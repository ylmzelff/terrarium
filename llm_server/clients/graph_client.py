"""
Microsoft Graph API Client for Outlook/Teams Integration.

This module provides a production-ready client for Microsoft Graph API,
enabling real-world calendar availability checks and Teams meeting creation.

Features:
- OAuth2 authentication with MSAL (Microsoft Authentication Library)
- Get user availability from Outlook calendars
- Create Teams online meetings
- Automatic token refresh
- Rate limiting (2 requests/second best practice)
- Timezone support with pytz

Usage:
    # Initialize client
    client = GraphAPIClient(
        client_id="your-azure-app-id",
        client_secret="your-secret",
        tenant_id="your-tenant-id"
    )
    
    # Authenticate
    client.authenticate(username="user@example.com", password="password")
    
    # Get availability
    availability = client.get_availability(
        email="user@example.com",
        start_datetime=datetime.now(),
        end_datetime=datetime.now() + timedelta(days=7)
    )
    
    # Create meeting
    meeting = client.create_teams_meeting(
        subject="Project Sync",
        start_datetime=datetime(2026, 2, 21, 10, 0),
        end_datetime=datetime(2026, 2, 21, 11, 0),
        attendees=["user1@example.com", "user2@example.com"]
    )

Requirements:
    pip install msal requests pytz

Azure Setup (Optional - for production use):
    1. Visit https://portal.azure.com
    2. Create App Registration in Azure Active Directory
    3. Add permissions: Calendars.Read, Calendars.ReadWrite, OnlineMeetings.ReadWrite
    4. Create client secret
    5. Save: client_id, client_secret, tenant_id

Education Accounts:
    - University M365 accounts work out-of-the-box!
    - Free M365 Developer Program: https://developer.microsoft.com/microsoft-365/dev-program
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import wraps

try:
    from msal import PublicClientApplication, SerializableTokenCache
    import requests
    import pytz
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning(
        "Microsoft Graph API dependencies not installed. "
        "Run: pip install msal requests pytz"
    )


logger = logging.getLogger(__name__)


def rate_limit(calls_per_second: float = 2.0):
    """
    Rate limiting decorator for Graph API calls.
    
    Microsoft Graph API limits:
    - 2000 requests per minute per app
    - Best practice: 2 requests/second to avoid throttling
    
    Args:
        calls_per_second: Maximum calls per second (default: 2.0)
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
                time.sleep(sleep_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator


class GraphAPIClient:
    def get_events(self, email: str, start_datetime: datetime, end_datetime: datetime, timezone: Optional[str] = None) -> list:
        """
        Belirtilen kullanıcı için Outlook/Teams takviminden etkinlikleri çeker.
        Args:
            email: Kullanıcı e-posta adresi
            start_datetime: Başlangıç zamanı (datetime)
            end_datetime: Bitiş zamanı (datetime)
            timezone: Zaman dilimi (opsiyonel)
        Returns:
            Etkinlik listesi (her biri bir dict)
        """
        self._ensure_valid_token()
        tz = timezone or self.timezone
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Prefer": f'outlook.timezone="{tz}"'
        }
        params = {
            "startDateTime": start_datetime.isoformat(),
            "endDateTime": end_datetime.isoformat()
        }
        url = f"{self.GRAPH_API_BASE}/users/{email}/calendarView"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("value", [])
    """
    Microsoft Graph API client for calendar and meeting operations.
    
    This client handles:
    - OAuth2 authentication via MSAL
    - Availability queries (getSchedule API)
    - Teams meeting creation
    - Token management and refresh
    - Rate limiting and error handling
    """
    
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
    SCOPES = ["Calendars.Read", "User.Read"]
    
    def __init__(
        self,
        client_id: str,
        client_secret: Optional[str] = None,
        tenant_id: str = "common",
        timezone: str = "UTC",
        token_cache_path: str = "token_cache.bin",
        fresh_auth: bool = True
    ):
        """
        Initialize Graph API client.
        
        Args:
            client_id: Azure App Registration client ID
            client_secret: Not used for public client flow but kept for compatibility
            tenant_id: Azure Active Directory tenant ID (default: 'common')
            timezone: Default timezone for operations (default: "UTC")
            token_cache_path: Path to token cache file
            fresh_auth: If True (default), clear all cached accounts at startup so
                        device flow is always triggered — same behaviour as test_graph_api.py.
                        Set to False to silently reuse cached tokens across runs.
        """
        if not GRAPH_AVAILABLE:
            raise ImportError(
                "Microsoft Graph API dependencies not installed. "
                "Run: pip install msal requests pytz"
            )
        
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.timezone = timezone
        self.token_cache_path = token_cache_path
        self.fresh_auth = fresh_auth
        
        # Initialize Token Cache
        self.cache = SerializableTokenCache()
        if os.path.exists(self.token_cache_path):
            with open(self.token_cache_path, "r") as f:
                self.cache.deserialize(f.read())
        
        # Initialize MSAL app (Public Client for Device Code flow)
        # Always use 'common' authority so personal Microsoft accounts (including
        # Gmail-linked Outlook accounts) can authenticate. A specific tenant_id
        # only works for work/school (M365) accounts.
        authority_url = "https://login.microsoftonline.com/common"
        self.app = PublicClientApplication(
            client_id=client_id,
            authority=authority_url,
            token_cache=self.cache
        )
        
        # Mirror test_graph_api.py: remove all cached accounts so device flow
        # is always triggered for a fresh interactive sign-in.
        if self.fresh_auth:
            for account in self.app.get_accounts():
                self.app.remove_account(account)
            logger.info("fresh_auth=True: cleared token cache — device flow will be triggered.")
        
        # We'll map email -> access_token in memory for rapid use during a session
        # but MSAL cache handles the actual persistence
        self.access_tokens: Dict[str, str] = {}
        
        logger.info(f"Graph API client initialized for tenant: {tenant_id}")

    
    def _save_cache(self):
        """Save the token cache to disk."""
        if self.cache.has_state_changed:
            with open(self.token_cache_path, "w") as f:
                f.write(self.cache.serialize())

    def get_token_for_user(self, email: str = None) -> str:
        """
        Get access token for a specific user using MSAL cache or Device Flow.
        
        Args:
            email: Provide a hint for the account to fetch. If it's a new account,
                   device flow will be triggered.
                   
        Returns:
            Access token string.
        """
        accounts = self.app.get_accounts()
        
        # If email is provided, try to find the matching account in cache
        target_account = None
        if email and accounts:
            for acc in accounts:
                if acc.get("username", "").lower() == email.lower():
                    target_account = acc
                    break
        
        # If no specific email matched, but we have exactly one account and no email requested, use it
        if not target_account and accounts and not email:
            target_account = accounts[0]
            
        if target_account:
            result = self.app.acquire_token_silent(self.SCOPES, account=target_account)
            if result and "access_token" in result:
                self._save_cache()
                if email:
                    self.access_tokens[email] = result["access_token"]
                return result["access_token"]
        
        # Fallback to Device Code Flow
        logger.info(f"No valid token in cache for {email}. Starting Device Code Flow...")
        print(f"\n[{email or 'New User'}] Authentication Required:")
        flow = self.app.initiate_device_flow(scopes=self.SCOPES)
        
        if "user_code" not in flow:
            raise Exception(f"Failed to create device flow. Err: {flow}")
            
        print(f"\n{flow['message']}\n")
        logger.info(f"Waiting for user to authenticate via device flow...")
        
        result = self.app.acquire_token_by_device_flow(flow)
        
        if "access_token" in result:
            logger.info(f"✅ Authentication successful for {email or 'User'}!")
            self._save_cache()
            
            # Fetch user email to store properly if not provided
            token = result["access_token"]
            
            # To be safe and strict, let's just return it
            if email:
                self.access_tokens[email] = token
            return token
            
        else:
            error_msg = result.get('error_description', result.get('error', 'Unknown error'))
            logger.error(f"❌ Authentication failed: {error_msg}")
            raise Exception(f"Authentication failed: {error_msg}")

    def authenticate(self, username: str, password: str) -> bool:
        # Deprecated: Kept for backwards compatibility but we shouldn't use it
        logger.warning("authenticate(username, password) is deprecated. Use get_token_for_user() instead.")
        token = self.get_token_for_user(email=username)
        return bool(token)
        
    def _ensure_valid_token(self, email: str = None) -> str:
        """Ensure access token is valid, refresh if needed. Returns the token."""
        # For backwards compatibility with other methods
        return self.get_token_for_user(email)
    
    @rate_limit(calls_per_second=2.0)
    def get_availability(
        self,
        email: str,
        start_datetime: datetime,
        end_datetime: datetime,
        timezone: Optional[str] = None,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Get user availability (free/busy schedule) from Outlook calendar.
        
        Uses Microsoft Graph getSchedule API:
        https://docs.microsoft.com/graph/api/calendar-getschedule
        
        Args:
            email: User email address
            start_datetime: Start of availability query window
            end_datetime: End of availability query window
            timezone: Timezone for query (default: instance timezone)
            interval_minutes: Slot duration in minutes (default: 60)
        
        Returns:
            List of availability slots:
            [
                {
                    "start": "2026-02-21T09:00:00",
                    "end": "2026-02-21T10:00:00",
                    "status": "free"  # or "busy", "tentative", "oof" (out of office)
                },
                ...
            ]
        
        Raises:
            Exception: If API call fails
        """
        token = self._ensure_valid_token(email)
        
        tz = timezone or self.timezone
        logger.info(f"Fetching availability for {email} ({start_datetime} to {end_datetime}, {tz})")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": f'outlook.timezone="{tz}"'
        }
        
        # Use calendarView instead of getSchedule:
        # - getSchedule only works for M365 work/school accounts
        # - calendarView works for personal accounts (Gmail-linked, Outlook.com, etc.)
        params = {
            "startDateTime": start_datetime.isoformat(),
            "endDateTime": end_datetime.isoformat(),
            "$select": "subject,start,end,showAs",
            "$orderby": "start/dateTime",
            "$top": 100
        }
        
        try:
            response = requests.get(
                f"{self.GRAPH_API_BASE}/me/calendarView",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            raw_events = data.get("value", [])

            # ── HAM VERİ LOGU ──────────────────────────────────────────────
            logger.info("=" * 70)
            logger.info(f"📥 HAM TEAMS/OUTLOOK VERİSİ — {len(raw_events)} etkinlik bulundu")
            logger.info(f"   Sorgu: {start_datetime.isoformat()} → {end_datetime.isoformat()}")
            logger.info(f"   Timezone: {tz}")
            if raw_events:
                for i, ev in enumerate(raw_events, 1):
                    subj      = ev.get("subject", "(başlıksız)")
                    ev_start  = ev.get("start", {}).get("dateTime", "?")
                    ev_end    = ev.get("end",   {}).get("dateTime", "?")
                    show_as   = ev.get("showAs", "?")
                    logger.info(f"   [{i:02d}] '{subj}'")
                    logger.info(f"        Başlangıç : {ev_start}")
                    logger.info(f"        Bitiş     : {ev_end}")
                    logger.info(f"        Durum     : {show_as}")
            else:
                logger.info("   (Bu aralıkta hiç takvim etkinliği yok)")
            logger.info("=" * 70)
            # ───────────────────────────────────────────────────────────────

            availability = self._parse_calendar_view_response(
                data, start_datetime, end_datetime, interval_minutes
            )
            
            logger.info(f"✅ Retrieved {len(availability)} availability slots")
            return availability
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Graph API error: {e}")
            raise Exception(f"Failed to fetch availability: {e}")
    
    def _parse_availability_response(
        self,
        response: Dict[str, Any],
        start_datetime: datetime,
        interval_minutes: int
    ) -> List[Dict[str, Any]]:
        """
        LEGACY: Parse Graph API getSchedule response.
        Kept for backwards compatibility. New code uses _parse_calendar_view_response.
        """
        slots = []
        if "value" not in response or not response["value"]:
            return slots
        schedule = response["value"][0]
        availability_view = schedule.get("availabilityView", "")
        status_map = {'0': 'free', '1': 'tentative', '2': 'busy', '3': 'oof', '4': 'working_elsewhere'}
        for i, status_code in enumerate(availability_view):
            slot_start = start_datetime + timedelta(minutes=i * interval_minutes)
            slot_end = slot_start + timedelta(minutes=interval_minutes)
            slots.append({
                "start": slot_start.isoformat(),
                "end": slot_end.isoformat(),
                "status": status_map.get(status_code, 'unknown')
            })
        return slots

    def _parse_calendar_view_response(
        self,
        response: Dict[str, Any],
        start_datetime: datetime,
        end_datetime: datetime,
        interval_minutes: int,
        work_hours_start: int = 9,
        work_hours_end: int = 18
    ) -> List[Dict[str, Any]]:
        """
        Parse Graph API calendarView response into fixed-interval availability slots.

        Only slots that fall WITHIN working hours (work_hours_start..work_hours_end)
        are included. Slots outside working hours are always marked 'busy' so that
        midnight-08:59 and 18:00+ don't pollute the availability array with fake 1s.

        Args:
            response: Raw calendarView API response (has 'value' list of events)
            start_datetime: Query start time
            end_datetime: Query end time
            interval_minutes: Slot duration in minutes
            work_hours_start: First working hour of day, inclusive (default: 9 → 09:00)
            work_hours_end: Last working hour of day, exclusive (default: 18 → until 18:00)

        Returns:
            List of dicts: [{"start": ..., "end": ..., "status": "free"|"busy"}, ...]
            Only work-hour slots are returned; off-hours slots are skipped entirely.
        """
        events = response.get("value", [])

        # Parse event times into (start, end) pairs as naive datetimes
        busy_intervals = []
        for event in events:
            show_as = event.get("showAs", "busy")  # free/tentative/busy/oof/workingElsewhere
            if show_as == "free":
                continue  # transparent/all-day free events — skip
            try:
                ev_start_str = event["start"]["dateTime"]
                ev_end_str   = event["end"]["dateTime"]
                ev_start = datetime.fromisoformat(ev_start_str.rstrip("Z").split(".")[0])
                ev_end   = datetime.fromisoformat(ev_end_str.rstrip("Z").split(".")[0])
                busy_intervals.append((ev_start, ev_end))
            except (KeyError, ValueError):
                continue

        logger.info(f"🔄 Slot dönüşümü başlıyor | Çalışma saatleri: {work_hours_start:02d}:00–{work_hours_end:02d}:00 | "
                    f"Meşgul aralıklar: {len(busy_intervals)}")

        # ── BUGFIX: Start from midnight of the FIRST day, not from datetime.now() ──
        # This ensures clean 00:00, 01:00, 02:00 ... 23:00 boundaries every day.
        # Non-work hours are marked 'busy' (not skipped), so the total slot count
        # always equals (num_days × 24) and matches slots_per_day=24 in the env config.
        naive_start = start_datetime.replace(tzinfo=None) if start_datetime.tzinfo else start_datetime
        naive_end   = end_datetime.replace(tzinfo=None)   if end_datetime.tzinfo   else end_datetime

        # Snap to midnight of first day
        day_start = naive_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = naive_end.replace  (hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        slot_start = day_start
        slots      = []
        slot_idx   = 0

        while slot_start < day_end:
            slot_end = slot_start + timedelta(minutes=interval_minutes)
            hour     = slot_start.hour

            # Mesai dışı saatler otomatik MEŞGUL
            outside_work = (hour < work_hours_start or hour >= work_hours_end)

            if outside_work:
                slots.append({
                    "start":  slot_start.isoformat(),
                    "end":    slot_end.isoformat(),
                    "status": "busy"
                })
                logger.debug(f"   Slot[{slot_idx:03d}] {slot_start.strftime('%Y-%m-%d %H:%M')}–{slot_end.strftime('%H:%M')} → ⬛ MESAİ DIŞI (0)")
            else:
                # Takvim etkinliği ile çakışıyor mu?
                is_busy = any(
                    ev_s < slot_end and ev_e > slot_start
                    for ev_s, ev_e in busy_intervals
                )
                icon = "🔴 MEŞGUL (0)" if is_busy else "🟢 MÜSAİT (1)"
                logger.info(f"   Slot[{slot_idx:03d}] {slot_start.strftime('%Y-%m-%d %H:%M')}–{slot_end.strftime('%H:%M')} → {icon}")
                slots.append({
                    "start":  slot_start.isoformat(),
                    "end":    slot_end.isoformat(),
                    "status": "busy" if is_busy else "free"
                })

            slot_start = slot_end
            slot_idx  += 1

        # Summary — only count work-hour slots for clarity
        work_slots  = [s for s in slots if work_hours_start <= datetime.fromisoformat(s["start"]).hour < work_hours_end]
        binary_work = [0 if s["status"] == "busy" else 1 for s in work_slots]
        binary_all  = [0 if s["status"] == "busy" else 1 for s in slots]
        logger.info("─" * 70)
        logger.info(f"📊 SLOT DÖNÜŞÜM ÖZET: {len(slots)} toplam slot | İş saati slotları: {len(work_slots)}")
        logger.info(f"   İş saati binary → {binary_work}")
        logger.info(f"   Tüm array (120)  → {binary_all}")
        logger.info("─" * 70)

        return slots
    
    @rate_limit(calls_per_second=2.0)
    def create_teams_meeting(
        self,
        subject: str,
        start_datetime: datetime,
        end_datetime: datetime,
        attendees: List[str],
        timezone: Optional[str] = None,
        body: Optional[str] = None,
        organizer_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Teams online meeting in Outlook calendar.
        
        Uses Microsoft Graph events API with isOnlineMeeting flag:
        https://docs.microsoft.com/graph/api/user-post-events
        
        Args:
            subject: Meeting title
            start_datetime: Meeting start time
            end_datetime: Meeting end time
            attendees: List of attendee email addresses
            timezone: Timezone for meeting (default: instance timezone)
            body: Meeting description/body (optional)
        
        Returns:
            Meeting details including Teams join URL:
            {
                "id": "event-id",
                "subject": "Project Sync",
                "start": "2026-02-21T10:00:00",
                "end": "2026-02-21T11:00:00",
                "join_url": "https://teams.microsoft.com/l/meetup/...",
                "attendees": ["user1@example.com", "user2@example.com"]
            }
        
        Raises:
            Exception: If API call fails
        """
        token = self._ensure_valid_token(organizer_email)
        
        tz = timezone or self.timezone
        logger.info(f"Creating Teams meeting: {subject} ({start_datetime} to {end_datetime})")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body or f"Meeting scheduled via Terrarium"
            },
            "start": {
                "dateTime": start_datetime.isoformat(),
                "timeZone": tz
            },
            "end": {
                "dateTime": end_datetime.isoformat(),
                "timeZone": tz
            },
            "attendees": [
                {
                    "emailAddress": {"address": email},
                    "type": "required"
                }
                for email in attendees
            ],
            "isOnlineMeeting": True,
            "onlineMeetingProvider": "teamsForBusiness"
        }
        
        try:
            response = requests.post(
                f"{self.GRAPH_API_BASE}/me/events",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            event = response.json()
            
            # Extract key details
            meeting_info = {
                "id": event.get("id"),
                "subject": event.get("subject"),
                "start": event["start"]["dateTime"],
                "end": event["end"]["dateTime"],
                "join_url": event.get("onlineMeeting", {}).get("joinUrl"),
                "attendees": attendees,
                "web_link": event.get("webLink")
            }
            
            logger.info(f"✅ Teams meeting created: {meeting_info['join_url']}")
            return meeting_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to create meeting: {e}")
            raise Exception(f"Failed to create Teams meeting: {e}")
    
    def get_events(self, email: str, start_datetime: datetime, end_datetime: datetime, timezone: Optional[str] = None) -> list:
        token = self._ensure_valid_token(email)
        tz = timezone or self.timezone
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": f'outlook.timezone="{tz}"'
        }
        params = {
            "startDateTime": start_datetime.isoformat(),
            "endDateTime": end_datetime.isoformat()
        }
        url = f"{self.GRAPH_API_BASE}/users/{email}/calendarView"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("value", [])
    
    @classmethod
    def from_env(cls, timezone: str = "UTC", fresh_auth: bool = True) -> "GraphAPIClient":
        """
        Create client from environment variables.
        
        Required environment variables:
        - AZURE_CLIENT_ID: App registration client ID
        - AZURE_CLIENT_SECRET: App registration secret
        - AZURE_TENANT_ID: Azure AD tenant ID
        
        Args:
            timezone: Default timezone (default: "UTC")
            fresh_auth: If True (default), always trigger device flow (clears cache).
        
        Returns:
            Initialized GraphAPIClient
        
        Raises:
            ValueError: If required env vars not set
        """
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        tenant_id = os.getenv("AZURE_TENANT_ID")
        
        if not all([client_id, client_secret, tenant_id]):
            raise ValueError(
                "Missing required environment variables: "
                "AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID"
            )
        
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            timezone=timezone,
            fresh_auth=fresh_auth
        )


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test if dependencies are installed
    if not GRAPH_AVAILABLE:
        print("❌ Missing dependencies. Install with:")
        print("   pip install msal requests pytz")
        exit(1)
    
    print("✅ Graph API client module loaded successfully")
    print("\nTo use this client:")
    print("1. Set environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")
    print("2. Create client: client = GraphAPIClient.from_env()")
    print("3. Authenticate: client.authenticate('user@example.com', 'password')")
    print("4. Get availability or create meetings")
