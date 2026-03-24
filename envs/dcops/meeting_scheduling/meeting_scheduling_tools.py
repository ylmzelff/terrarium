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

        # Strict phase gating prevents bypassing OT flow in execution phase.
        if phase:
            allowed_tool_names = {
                tool.get("function", {}).get("name")
                for tool in self.get_tools(phase)
                if tool.get("function", {}).get("name")
            }
            if allowed_tool_names and tool_name not in allowed_tool_names:
                return {
                    "status": "retry",
                    "reason": (
                        f"Tool '{tool_name}' is not allowed during phase '{phase}'. "
                        f"Allowed tools: {sorted(allowed_tool_names)}"
                    ),
                }

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
        Fetch the calling agent's calendar events.

        Two modes based on env_state["use_real_calendars"]:
          • False (simulation): Return pre-generated availability from env_state
          • True  (production): Fetch live data from Microsoft Graph API using
                                 credentials from the YAML config (graph_api block)
        """
        if not env_state:
            return {"error": "env_state not provided — cannot fetch calendar."}

        meeting_id = arguments.get("meeting_id")
        if not meeting_id:
            return {"error": "'meeting_id' is required."}

        num_days       = env_state.get("num_days", 5)
        slots_per_day  = env_state.get("slots_per_day", 24)
        total_slots    = num_days * slots_per_day
        use_real       = env_state.get("use_real_calendars", False)

        # ────────────────────────────────────────────────────────────────
        # SIMULATION MODE — no Graph API needed
        # ────────────────────────────────────────────────────────────────
        if not use_real:
            simulated = env_state.get("simulated_availability", {})
            meeting_avail = simulated.get(meeting_id, {})
            agent_slots = meeting_avail.get(agent_name)

            if agent_slots is None:
                return {
                    "error": (
                        f"No simulated availability found for agent '{agent_name}' "
                        f"in meeting '{meeting_id}'. Check config."
                    )
                }

            # Convert binary array to event-like structure for the LLM
            from datetime import datetime, timedelta
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            events = []
            for i, val in enumerate(agent_slots):
                slot_start = start_dt + timedelta(hours=i)
                slot_end   = slot_start + timedelta(hours=1)
                events.append({
                    "slot_index": i,
                    "start":  slot_start.isoformat(),
                    "end":    slot_end.isoformat(),
                    "status": "free" if val == 1 else "busy",
                })

            free_count = sum(agent_slots)
            logger.info(
                "🔬 [%s] SIMULATION MODE — %d/%d slots free | meeting=%s",
                agent_name, free_count, total_slots, meeting_id,
            )

            # ── Log fetch event to blackboard ──
            available_indices = [i for i, v in enumerate(agent_slots) if v == 1]
            try:
                self.blackboard_manager.post_system_message(
                    blackboard_id=0,
                    kind="agent_fetch_calendar",
                    payload={
                        "message": f"[STEP] {agent_name} fetched calendar for meeting {meeting_id}",
                        "agent": agent_name,
                        "meeting_id": meeting_id,
                        "mode": "simulation",
                        "total_slots": total_slots,
                        "free_slots": free_count,
                        "busy_slots": total_slots - free_count,
                        "generated_array": agent_slots,
                        "available_indices": available_indices,
                    },
                )
            except Exception as exc:
                logger.warning("Could not log fetch event to blackboard: %s", exc)

            # In simulation mode, array is already fully known (no parsing needed).
            # Auto-submit to environment via state_updates so OT can run as soon as
            # both participants have fetched their calendars.
            if meeting_id not in self._submitted_arrays:
                self._submitted_arrays[meeting_id] = {}
            self._submitted_arrays[meeting_id][agent_name] = list(agent_slots)

            meetings     = env_state.get("meetings", {})
            meeting_info = meetings.get(meeting_id, {})
            participants = meeting_info.get("participants", list(self._submitted_arrays[meeting_id].keys()))
            submitted    = self._submitted_arrays[meeting_id]
            waiting_for  = [p for p in participants if p not in submitted]

            state_updates = {
                "submitted_arrays": {
                    meeting_id: dict(submitted)
                }
            }

            if waiting_for:
                auto_submit_note = (
                    f"Simulation mode: your generated array was auto-submitted. "
                    f"Waiting for {waiting_for} before OT runs."
                )
                auto_submit_status = "waiting_for_other_agent"
            else:
                auto_submit_note = (
                    "Simulation mode: both arrays are now submitted. "
                    "OT intersection computation has been triggered."
                )
                auto_submit_status = "all_submitted_ot_triggered"

            return {
                "meeting_id": meeting_id,
                "agent":      agent_name,
                "events":     events,
                "slot_info": {
                    "total_slots":           total_slots,
                    "slots_per_day":         slots_per_day,
                    "num_days":              num_days,
                    "slot_duration_minutes": 60,
                    "date_range":            f"{start_dt.date()} to {(start_dt + timedelta(days=num_days - 1)).date()}",
                    "work_hours":            "09:00–18:00 (slots outside this range are marked busy)",
                },
                "instructions": (
                    f"Build a binary array of EXACTLY {total_slots} integers (0 or 1).\n"
                    "  • 1 = you are FREE at that slot\n"
                    "  • 0 = you are BUSY at that slot\n"
                    "  • Slots where status='busy' → 0\n"
                    "  • All other slots → 1\n"
                    f"Then call: submit_availability_array(meeting_id='{meeting_id}', "
                    f"availability=[...{total_slots} values...])"
                ),
                "auto_submit": {
                    "enabled": True,
                    "status": auto_submit_status,
                    "message": auto_submit_note,
                    "waiting_for": waiting_for,
                },
                "state_updates": state_updates,
            }

        # ────────────────────────────────────────────────────────────────
        # PRODUCTION MODE — fetch RAW Teams/Outlook events from Graph API
        # The code ONLY fetches raw data. The LLM reads the events and
        # builds the binary availability array by itself.
        # Credentials come ONLY from YAML (graph_api block in env_state)
        # FAILS HARD: any error → RuntimeError → simulation stops
        # ────────────────────────────────────────────────────────────────
        graph_config   = env_state.get("graph_api", {})
        agent_emails   = graph_config.get("agent_emails", {})
        email          = agent_emails.get(agent_name)
        timezone       = graph_config.get("timezone", "Turkey Standard Time")

        if not email:
            raise RuntimeError(
                f"❌ SIMULATION STOPPED: No email configured for agent '{agent_name}'.\n"
                f"   Set graph_api.agent_emails.{agent_name} in the YAML config."
            )

        from datetime import datetime, timedelta
        try:
            from llm_server.clients.graph_client import GraphAPIClient
        except ImportError as exc:
            raise RuntimeError(
                "❌ SIMULATION STOPPED: Graph API dependencies missing.\n"
                "   Run: pip install msal requests pytz"
            ) from exc

        # Reuse or create the Graph API client (cached per tool instance)
        if not hasattr(self, "_graph_client") or self._graph_client is None:
            self._graph_client = GraphAPIClient.from_yaml(env_state)
            logger.info("✅ Graph API client initialised from YAML config.")

        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt   = start_dt + timedelta(days=num_days)

        # Fetch RAW calendar data — code does NOT compute slots
        logger.info("📅 [%s] fetch_raw_calendar | %s | %s → %s",
                    agent_name, email, start_dt.date(), end_dt.date())
        raw_data = self._graph_client.fetch_raw_calendar(
            email=email,
            start_datetime=start_dt,
            end_datetime=end_dt,
            timezone=timezone,
        )

        raw_events = raw_data.get("events", [])
        user_info  = raw_data.get("user", {})

        logger.info("📋 [%s] %d raw events fetched from Teams/Outlook | meeting=%s",
                    agent_name, len(raw_events), meeting_id)

        # -------------------------------------------------------------------
        # Pretty console print for debugging (so the user can see what the LLM sees)
        # -------------------------------------------------------------------
        print(f"\n{'='*80}")
        print(f"🗓️  [{agent_name}] CALENDAR EVENTS FETCHED FROM GRAPH API")
        print(f"{'-'*80}")
        if not raw_events:
            print("   (No events found in this date range)")
        else:
            for i, evt in enumerate(raw_events):
                subj   = evt.get('subject', 'No Subject') or 'No Subject'
                start  = evt.get('start', '')
                end    = evt.get('end', '')
                showAs = str(evt.get('showAs', '')).upper()
                start_dt_str = start[:16].replace("T", " ") if len(start) >= 16 else start
                end_dt_str   = end[:16].replace("T", " ") if len(end) >= 16 else end
                print(f"   {i+1:02d}. {start_dt_str} → {end_dt_str} | {showAs:9} | {subj}")
        print(f"{'='*80}\n")


        return {
            "meeting_id": meeting_id,
            "agent":      agent_name,
            "user":       user_info,
            "raw_events": raw_events,
            "slot_info": {
                "total_slots":           total_slots,
                "slots_per_day":         slots_per_day,
                "num_days":              num_days,
                "slot_duration_minutes": 60,
                "date_range":            f"{start_dt.date()} to {(end_dt - timedelta(days=1)).date()}",
            },
            "instructions": (
                "You received your RAW calendar events from Microsoft Teams/Outlook.\n"
                "Now YOU must build a binary availability array.\n\n"
                "RULES:\n"
                f"  • Array length MUST be EXACTLY {total_slots} (= {num_days} days × {slots_per_day} slots/day)\n"
                "  • Each slot = 1 hour, starting from 00:00 of the first day\n"
                "  • Slot 0 = Day1 00:00–01:00, Slot 1 = Day1 01:00–02:00, ..., Slot 9 = Day1 09:00–10:00, etc.\n"
                "  • Work hours: 09:00–18:00 (slots outside this = 0)\n\n"
                "HOW TO BUILD THE ARRAY:\n"
                "  1. Start with all slots = 0\n"
                "  2. For slots within 09:00–18:00 work hours: set to 1 (available)\n"
                "  3. For each event in raw_events where showAs='busy' or 'tentative':\n"
                "     Find which slot(s) overlap with the event time → set those to 0\n"
                "  4. Slots outside 09:00–18:00 remain 0\n\n"
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
        available_indices = [i for i, v in enumerate(availability) if v == 1]
        logger.info(
            "📥 Availability submitted | meeting=%s | agent=%s | free=%d/%d",
            meeting_id, agent_name, free_count, total_slots,
        )

        # -------------------------------------------------------------------
        # Pretty console print to show the matrix the LLM built
        # -------------------------------------------------------------------
        print(f"\n{'='*80}")
        print(f"🧠 [{agent_name}] LLM GENERATED BINARY ARRAY (Meeting: {meeting_id})")
        print(f"{'-'*80}")
        # Print a header row like 00 01 02 ... 23
        header = "      " + " ".join(f"{hr:02d}" for hr in range(min(slots_per_day, 24)))
        print(header)
        
        # Determine actual days based on array length just in case
        actual_days = len(availability) // slots_per_day
        for day in range(actual_days):
            s_idx = day * slots_per_day
            e_idx = s_idx + slots_per_day
            day_slots = availability[s_idx:e_idx]
            slot_str  = "  ".join(str(x) for x in day_slots)
            print(f"Day {day+1}: {slot_str}")
            
        print(f"{'='*80}\n")

        # ── Log submit event to blackboard ──
        try:
            self.blackboard_manager.post_system_message(
                blackboard_id=0,
                kind="agent_submit_array",
                payload={
                    "message": f"[STEP] {agent_name} submitted availability array for meeting {meeting_id}",
                    "agent": agent_name,
                    "meeting_id": meeting_id,
                    "total_slots": total_slots,
                    "free_slots": free_count,
                    "busy_slots": total_slots - free_count,
                    "submitted_array": availability,
                    "available_indices": available_indices,
                },
            )
        except Exception as exc:
            logger.warning("Could not log submit event to blackboard: %s", exc)

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

        allowed             = self._allowed_intervals(start, end)
        normalized_interval = self._normalize_interval(interval, start, end)

        if normalized_interval not in allowed:
            sample = ", ".join(sorted(allowed)[:5])
            return {
                "status": "retry",
                "reason": f"interval '{interval}' not valid for meeting window [{start},{end}).",
                "suggestions": [f"Valid examples: {sample}"],
            }

        # Enforce OT result usage: if intersection exists, agent must pick the earliest
        # common slot for that meeting.
        meeting_intersections = env_state.get("meeting_intersections", {}) if env_state else {}
        intersection = meeting_intersections.get(meeting_id)
        if intersection is not None:
            common_slots = [i for i, v in enumerate(intersection) if v == 1]
            if common_slots:
                chosen_start = int(normalized_interval.split("-")[0])
                earliest = common_slots[0]
                if chosen_start != earliest:
                    return {
                        "status": "retry",
                        "reason": (
                            f"You must choose the earliest OT intersection slot. "
                            f"Earliest={earliest}, but got {chosen_start}."
                        ),
                        "suggestions": [f"Use interval='{earliest}'"],
                    }
            else:
                return {
                    "status": "retry",
                    "reason": (
                        "OT intersection has no common slot for this meeting. "
                        "Do not schedule attendance for this meeting."
                    ),
                }
        else:
            return {
                "status": "retry",
                "reason": (
                    "OT intersection is not available yet for this meeting. "
                    "Complete planning (array submission/OT) before attend_meeting."
                ),
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
                    "window": [start, end],
                    "participants": participants,
                },
                "slot_index": int(str(interval).strip().split("-")[0]),
                "interval": normalized_interval,
                "total_assigned": len(updated_attendance),
                "remaining_variables": int(total_vars) - len(updated_attendance),
                "state_updates": {"attendance": updated_attendance},
            },
        }

        # Extract raw slot index for clear logging
        raw_slot = str(interval).strip().split("-")[0]
        action = {
            "action": "attend_meeting",
            "meeting_id": meeting_id,
            "slot_index": int(raw_slot) if raw_slot.isdigit() else raw_slot,
            "interval": interval,
        }
        if self.blackboard_manager:
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, result, phase, iteration
            )

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _allowed_intervals(start: int, end: int) -> Set[str]:
        allowed: Set[str] = {"skip"}
        for join in range(start, end):
            for leave in range(join + 1, end + 1):
                allowed.add(f"{join}-{leave}")
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
