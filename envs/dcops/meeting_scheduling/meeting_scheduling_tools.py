"""
Meeting Scheduling — Tool Definitions & Handlers
=================================================
Planning Phase Tools (new — agentic flow):
  • fetch_my_calendar      : Agent fetches its own calendar events from Graph API
  • submit_availability_array : Agent submits its binary availability; triggers OT when both ready

Execution Phase Tool (existing):
  • attend_meeting         : Agent commits to a specific time slot
"""

import logging
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool JSON Schemas
# ─────────────────────────────────────────────────────────────────────────────

_FETCH_MY_CALENDAR_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_my_calendar",
        "description": (
            "Fetch YOUR OWN calendar events from Microsoft Outlook / Teams. "
            "Returns raw events and slot configuration so you can build your "
            "binary availability array. Call this FIRST in the planning phase."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The meeting ID you are fetching availability for (e.g. 'm001').",
                },
            },
            "required": ["meeting_id"],
        },
    },
}

_SUBMIT_AVAILABILITY_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_availability_array",
        "description": (
            "Submit your binary availability array after building it from your calendar events. "
            "1 = free/available, 0 = busy/unavailable. "
            "When both participants have submitted, the OT (Oblivious Transfer) protocol runs "
            "automatically and the intersection is posted to the blackboard."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The meeting ID this availability is for (e.g. 'm001').",
                },
                "availability": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [0, 1]},
                    "description": (
                        "Binary array of length equal to total_slots (e.g. 120 for 5 days × 24 slots). "
                        "1 = available, 0 = busy. "
                        "Mark slots outside working hours (00:00-09:00, 18:00-24:00) as 0."
                    ),
                },
            },
            "required": ["meeting_id", "availability"],
        },
    },
}

_ATTEND_MEETING_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "attend_meeting",
        "description": "Choose your attendance interval for a meeting you participate in.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "Meeting ID from your meeting list (e.g., m001).",
                },
                "interval": {
                    "type": "string",
                    "description": (
                        "The slot number to attend (e.g. '0' for the first slot). "
                        "Use the SMALLEST index where the intersection = 1."
                    ),
                },
            },
            "required": ["meeting_id", "interval"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MeetingSchedulingTools
# ─────────────────────────────────────────────────────────────────────────────

class MeetingSchedulingTools:
    """
    Environment tools for the MeetingScheduling scenario.

    Planning phase (agentic flow):
      1. fetch_my_calendar       → Agent fetches its own calendar
      2. submit_availability_array → Agent submits binary array; OT triggers automatically

    Execution phase:
      3. attend_meeting          → Agent commits to earliest common slot
    """

    def __init__(self, blackboard_manager, env=None):
        self.blackboard_manager = blackboard_manager
        # env is kept for optional direct-call path (not used in MCP mode)
        self.env = env
        # Server-side state: submitted availability arrays awaiting OT
        # { meeting_id: { agent_name: [0, 1, ...] } }
        self._submitted_arrays: Dict[str, Dict[str, List[int]]] = {}

    # ── Tool Discovery ─────────────────────────────────────────────────────

    def get_tool_names(self) -> Set[str]:
        return {"fetch_my_calendar", "submit_availability_array", "attend_meeting"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase == "planning":
            return [_FETCH_MY_CALENDAR_SCHEMA, _SUBMIT_AVAILABILITY_SCHEMA]
        if phase == "execution":
            return [_ATTEND_MEETING_SCHEMA]
        return []

    # ── Unified Tool Call Router ───────────────────────────────────────────

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("🛠  Tool dispatch | agent=%s | tool=%s | phase=%s", agent_name, tool_name, phase)

        if tool_name == "fetch_my_calendar":
            return self._handle_fetch_my_calendar(agent_name, arguments, env_state)
        if tool_name == "submit_availability_array":
            return self._handle_submit_availability_array(agent_name, arguments, env_state, phase, iteration)
        if tool_name == "attend_meeting":
            return self._handle_attend_meeting(agent_name, arguments, phase, iteration, env_state)

        return {"error": f"Unknown tool: '{tool_name}'"}

    # ─────────────────────────────────────────────────────────────────────
    # Planning Tool 1: fetch_my_calendar
    # ─────────────────────────────────────────────────────────────────────

    def _handle_fetch_my_calendar(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        env_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Fetch the calling agent's calendar events via Graph API.
        Uses env_state to get graph_api config (email, timezone, num_days, slots_per_day).
        """
        if not env_state:
            return {"error": "env_state not provided — cannot fetch calendar."}

        meeting_id = arguments.get("meeting_id")
        if not meeting_id:
            return {"error": "'meeting_id' is required."}

        # Pull config from env_state
        graph_config   = env_state.get("graph_api", {})
        agent_emails   = graph_config.get("agent_emails", {})
        email          = agent_emails.get(agent_name)
        num_days       = env_state.get("num_days", 5)
        slots_per_day  = env_state.get("slots_per_day", 24)
        timezone       = graph_config.get("timezone", "UTC")
        total_slots    = num_days * slots_per_day

        if not email:
            return {
                "error": f"No email configured for agent '{agent_name}'. "
                         f"Set graph_api.agent_emails.{agent_name} in the config."
            }

        from datetime import datetime, timedelta
        try:
            from llm_server.clients.graph_client import GraphAPIClient
        except ImportError as exc:
            return {"error": f"Graph API client not available: {exc}"}

        # Reuse or create the Graph API client (cached per tool instance)
        if not hasattr(self, "_graph_client") or self._graph_client is None:
            try:
                self._graph_client = GraphAPIClient.from_env(timezone=timezone)
                logger.info("✅ Graph API client initialised in MeetingSchedulingTools.")
            except Exception as exc:
                logger.exception("Failed to init Graph API client: %s", exc)
                return {"error": f"Graph API client init failed: {exc}"}

        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt   = start_dt + timedelta(days=num_days)

        try:
            logger.info("📅 [%s] fetch_my_calendar | %s | %s → %s",
                        agent_name, email, start_dt.date(), end_dt.date())
            raw_slots = self._graph_client.get_availability(
                email=email,
                start_datetime=start_dt,
                end_datetime=end_dt,
                interval_minutes=60,
            )
        except Exception as exc:
            logger.exception("Graph API call failed for %s: %s", agent_name, exc)
            return {"error": f"Calendar fetch failed: {exc}"}

        events = [
            {
                "slot_index": i,
                "start":  s["start"],
                "end":    s["end"],
                "status": s["status"],
            }
            for i, s in enumerate(raw_slots)
        ]

        busy_count = sum(1 for e in events if e["status"] == "busy")
        logger.info("📋 [%s] %d slots returned | %d busy | meeting=%s",
                    agent_name, len(events), busy_count, meeting_id)

        return {
            "meeting_id": meeting_id,
            "agent":      agent_name,
            "events":     events,
            "slot_info": {
                "total_slots":           total_slots,
                "slots_per_day":         slots_per_day,
                "num_days":              num_days,
                "slot_duration_minutes": 60,
                "date_range":            f"{start_dt.date()} to {(end_dt - timedelta(days=1)).date()}",
                "work_hours":            "09:00–18:00 (slots outside this range are marked busy)",
            },
            "instructions": (
                f"Build a binary array of EXACTLY {total_slots} integers (0 or 1).\n"
                "  • 1 = you are FREE at that slot\n"
                "  • 0 = you are BUSY at that slot\n"
                "  • Slots where status='busy' → 0\n"
                "  • Slots outside 09:00–18:00 work hours → 0\n"
                "  • All other slots → 1\n"
                f"Then call: submit_availability_array(meeting_id='{meeting_id}', "
                f"availability=[...{total_slots} values...])"
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Planning Tool 2: submit_availability_array
    # ─────────────────────────────────────────────────────────────────────

    def _handle_submit_availability_array(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        env_state: Optional[Dict[str, Any]],
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, Any]:
        """
        Store the agent's binary array server-side.
        Returns state_updates with submitted_arrays so sequential.py can propagate
        them back to the environment, which will trigger OT when all participants ready.
        """
        meeting_id   = arguments.get("meeting_id")
        availability = arguments.get("availability")

        if not meeting_id:
            return {"error": "'meeting_id' is required."}
        if availability is None:
            return {"error": "'availability' array is required."}
        if not isinstance(availability, list) or not all(v in (0, 1) for v in availability):
            return {"error": "'availability' must be a list of 0s and 1s."}

        # Validate length against env_state
        num_days      = env_state.get("num_days", 5) if env_state else 5
        slots_per_day = env_state.get("slots_per_day", 24) if env_state else 24
        total_slots   = num_days * slots_per_day

        if len(availability) != total_slots:
            return {
                "error": (
                    f"Expected {total_slots} values ({num_days} days × {slots_per_day} slots/day), "
                    f"got {len(availability)}. Rebuild your array with exactly {total_slots} values."
                )
            }

        # Store server-side
        if meeting_id not in self._submitted_arrays:
            self._submitted_arrays[meeting_id] = {}
        self._submitted_arrays[meeting_id][agent_name] = availability

        free_count = sum(availability)
        logger.info(
            "📥 Availability submitted | meeting=%s | agent=%s | free=%d/%d",
            meeting_id, agent_name, free_count, total_slots,
        )

        # Who are the participants for this meeting?
        meetings     = env_state.get("meetings", {}) if env_state else {}
        meeting_info = meetings.get(meeting_id, {})
        participants = meeting_info.get("participants", list(self._submitted_arrays[meeting_id].keys()))
        submitted    = self._submitted_arrays[meeting_id]
        waiting_for  = [p for p in participants if p not in submitted]

        # Propagate to env via state_updates regardless (env.apply_state_updates will handle OT)
        state_updates = {"submitted_arrays": {meeting_id: dict(submitted)}}

        if waiting_for:
            return {
                "status":       "received",
                "meeting_id":   meeting_id,
                "agent":        agent_name,
                "free_slots":   free_count,
                "waiting_for":  waiting_for,
                "message":      f"Array saved ({free_count}/{total_slots} free). Waiting for {waiting_for} to submit.",
                "state_updates": state_updates,
            }

        logger.info("🔒 All participants submitted → signalling env to run OT | meeting=%s", meeting_id)
        return {
            "status":        "all_submitted",
            "meeting_id":    meeting_id,
            "message":       (
                f"Both agents have submitted. OT protocol will now compute the intersection. "
                "Check the blackboard shortly for the common availability table."
            ),
            "state_updates": state_updates,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Execution Tool: attend_meeting  (unchanged logic, clean rewrite)
    # ─────────────────────────────────────────────────────────────────────

    def _handle_attend_meeting(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str],
        iteration: Optional[int],
        env_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not env_state:
            return {"status": "failed", "reason": "Environment state not provided."}

        meetings    = env_state.get("meetings", {})
        attendance  = env_state.get("attendance", {})
        agent_names = env_state.get("agent_names", [])

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent '{agent_name}' not found."}

        meeting_id = arguments.get("meeting_id")
        interval   = arguments.get("interval")

        if not meeting_id:
            return {"status": "retry", "reason": "'meeting_id' is required."}
        if interval is None:
            return {"status": "retry", "reason": "'interval' is required."}
        if meeting_id not in meetings:
            return {"status": "failed", "reason": f"Meeting '{meeting_id}' not found."}

        meeting      = meetings[meeting_id]
        participants = meeting.get("participants", [])

        if agent_name not in participants:
            return {
                "status": "failed",
                "reason": f"Agent '{agent_name}' is not a participant of '{meeting_id}'.",
            }

        var_name = f"{agent_name}__{meeting_id}"
        if var_name in attendance:
            return {"status": "failed", "reason": f"Attendance for '{meeting_id}' already set."}

        # Normalise interval: support "3" (slot index) or "3-4" (range)
        try:
            start = int(meeting.get("start", 0))
            end   = int(meeting.get("end", 0))
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "Invalid meeting window in state."}

        meeting_type        = str(meeting.get("meeting_type", "soft"))
        allowed             = self._allowed_intervals(start, end, meeting_type)
        normalized_interval = self._normalize_interval(interval, start, end)

        if normalized_interval not in allowed:
            sample = ", ".join(sorted(allowed)[:5])
            return {
                "status": "retry",
                "reason": f"interval '{interval}' not valid for meeting window [{start},{end}).",
                "suggestions": [f"Valid examples: {sample}"],
            }

        updated_attendance = dict(attendance)
        updated_attendance[var_name] = normalized_interval

        total_vars = env_state.get("total_variables") or sum(
            len(m.get("participants", [])) for m in meetings.values()
        )

        result = {
            "status": "success",
            "result": {
                "agent": agent_name,
                "meeting": {
                    "id": meeting_id,
                    "title": meeting.get("title"),
                    "meeting_type": meeting_type,
                    "window": [start, end],
                    "participants": participants,
                },
                "interval": normalized_interval,
                "total_assigned": len(updated_attendance),
                "remaining_variables": int(total_vars) - len(updated_attendance),
                "state_updates": {"attendance": updated_attendance},
            },
        }

        action = {"action": "attend_meeting", "meeting_id": meeting_id, "interval": interval}
        if self.blackboard_manager:
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, result, phase, iteration
            )

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _allowed_intervals(start: int, end: int, meeting_type: str) -> Set[str]:
        allowed: Set[str] = {"skip"}
        for join in range(start, end):
            for leave in range(join + 1, end + 1):
                allowed.add(f"{join}-{leave}")
            if meeting_type == "strict":
                break
        return allowed

    @staticmethod
    def _normalize_interval(interval: str, start: int, end: int) -> str:
        """Convert various interval formats to canonical 'start-end' form."""
        interval = str(interval).strip()
        if interval.isdigit():
            # Agent sent a slot index (0-indexed)
            idx = int(interval)
            if start <= idx < end:
                return f"{idx}-{idx + 1}"
            return f"{idx}-{idx + 1}"  # let validation catch out-of-bounds
        return interval
