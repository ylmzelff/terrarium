from typing import Dict, Any, List, Optional

from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import build_vllm_tool_instructions, get_phase_tool_instructions

# sana verilen 1 dimensional arraylardan ortak intersection bul(1 olan slotlar için) ve executiondan sonra karar verilen index i döndür.
class MeetingSchedulingPrompts:
    """Prompt builder for the MeetingScheduling environment.

    Agents decide attendance intervals for each meeting they participate in.
    """

    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger(env.__class__.__name__, env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            planning_tool_lines=[
                "- fetch_my_calendar(meeting_id: str): "
                "Fetch YOUR OWN calendar events from Microsoft Teams/Outlook. "
                "Call this FIRST in planning phase to get your availability.",
                "- submit_availability_array(meeting_id: str, availability: list[int]): "
                "Submit your binary availability array (0=busy, 1=free). "
                "Call this AFTER building your array from calendar events.",
            ],
            execution_tool_lines=[
                "- attend_meeting(meeting_id: str, interval: str): "
                "Schedule your attendance at a specific time slot. "
                "interval should be the SLOT INDEX where intersection = 1 (e.g., '9' for slot index 9). "
                "Select the SMALLEST index where value = 1 in the meeting intersection row."
            ],
            planning_header="Planning phase tools (agentic calendar flow + blackboard):",
            execution_header="Execution phase tools (commit attendance decision):",
            system_note=(
                "Planning: fetch your calendar, build binary array, submit it, then coordinate via blackboard.\n"
                "Execution: attend_meeting to commit to the earliest common slot."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = """You are a fully autonomous agent in a meeting scheduling system.

YOUR GOAL: Schedule the meeting at the EARLIEST common available time slot.

PHASES:

=== PLANNING PHASE — AGENTIC CALENDAR FLOW ===
You must complete ALL 5 steps in order:

  STEP 1 → Call fetch_my_calendar(meeting_id)
           This returns your calendar events and slot configuration.

  STEP 2 → Read the returned events carefully.
           For each slot: if status='free' AND hour is 09:00-18:00 → mark as 1 (available)
                         otherwise → mark as 0 (busy)

  STEP 3 → Build a binary array of EXACTLY (num_days × slots_per_day) values.
           Example: 5 days × 24 slots = 120 values: [0,0,...,1,1,...]

  STEP 4 → Call submit_availability_array(meeting_id, availability=[your array])
           The system will run the privacy-preserving OT protocol automatically
           when both participants have submitted.

  STEP 5 → Coordinate via post_message on the shared blackboard.
           Once OT completes, the common availability table will appear in the blackboard.
           Agree on the EARLIEST (smallest index) slot where intersection = 1.

=== EXECUTION PHASE ===
  → Call attend_meeting(meeting_id, interval='SLOT_INDEX') with the agreed slot.
     Use the SMALLEST index where the meeting intersection row = 1.
     NEVER use interval='skip' if any common slots exist.

CRITICAL RULES:
  - You may ONLY fetch YOUR OWN calendar (fetch_my_calendar uses your agent identity)
  - Do NOT reveal your raw availability to the other agent (only OT result is shared)
  - Always respect the 5-step order in planning phase"""

        system_text = (self.tool_instruction_data or {}).get("system")
        if system_text:
            base_prompt += "\n\nTOOL CALLING REQUIREMENTS:\n" + system_text
        return base_prompt

    def get_user_prompt(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        system_prompt = self.get_system_prompt()
        user_prompt = self._get_user_prompt_impl(agent_name, agent_context, blackboard_context)

        if self.prompt_logger:
            self.prompt_logger.log_prompts(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                phase=agent_context.get("phase", "unknown"),
                iteration=agent_context.get("iteration"),
                round_num=agent_context.get("planning_round"),
            )

        return user_prompt

    def _lookup_meeting(self, meeting_id: str):
        for meeting in self.env.instance.meetings:
            if meeting.meeting_id == meeting_id:
                return meeting
        return None

    def _get_user_prompt_impl(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        phase = agent_context.get("phase", "unknown")
        iteration = agent_context.get("iteration", 0)

        context_parts: List[str] = [
            "=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            f"You are agent {agent_name}",
            "",
        ]

        # Agent-specific instruction from CoLLAB instance if available
        instruction: Optional[str] = getattr(self.env.instance, "explanations", {}).get(agent_name)
        if not instruction:
            try:
                instruction = self.env.problem.agent_instruction(agent_name)
            except Exception:
                instruction = None
        if instruction:
            context_parts.append("=== YOUR INSTRUCTIONS ===")
            context_parts.append(instruction)
            context_parts.append("")

        agent_vars = agent_context.get("agent_variables") or [
            var.name for var in self.env.problem.agent_variables(agent_name)
        ]

        context_parts.append("=== YOUR MEETINGS ===")
        for var_name in agent_vars:
            meeting_id = var_name.split("__", 1)[1] if "__" in var_name else var_name
            meeting = self._lookup_meeting(meeting_id)
            current_choice = self.env.assignment.get(var_name)
            status = f"CHOSEN: {current_choice}" if current_choice is not None else "PENDING"
            if meeting:
                context_parts.append(
                    f"- {meeting.meeting_id}: {meeting.title} "
                    f"participants: {', '.join(meeting.participants)} :: {status}"
                )
            else:
                context_parts.append(f"- {meeting_id} :: {status}")
        context_parts.append("")

        if self.env.assignment:
            context_parts.append("=== CURRENT JOINT ATTENDANCE (for coordination) ===")
            for var, val in sorted(self.env.assignment.items()):
                context_parts.append(f"{var}: {val}")
            context_parts.append("")

        if blackboard_context:
            context_parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    context_parts.append(f"Blackboard {bb_id}:")
                    context_parts.append(content)
                    context_parts.append("")

        if phase == "planning":
            context_parts.extend(
                [
                    "=== CURRENT PHASE: PLANNING ===",
                    "Follow these 5 steps IN ORDER:",
                    "",
                    "  [1] Call fetch_my_calendar(meeting_id=<your meeting id>)",
                    "      → You will receive your calendar events and slot configuration.",
                    "",
                    "  [2] Read the events. For each slot:",
                    "      • status='free' AND 09:00≤hour<18:00  → 1 (available)",
                    "      • anything else                        → 0 (busy)",
                    "",
                    "  [3] Build a binary array of EXACTLY (num_days × slots_per_day) integers.",
                    "      Example: 5 × 24 = 120 values: [0,0,...,1,1,...]",
                    "",
                    "  [4] Call submit_availability_array(meeting_id=..., availability=[...])",
                    "      → OT runs automatically once BOTH agents have submitted.",
                    "      → The intersection table will appear in the shared blackboard.",
                    "",
                    "  [5] Post a message to the blackboard with post_message:",
                    "      Tell the other agent which slot (smallest common index) you propose.",
                    "",
                ]
            )
        elif phase == "execution":
            context_parts.extend(
                [
                    "=== CURRENT PHASE: EXECUTION ===",
                    "Commit your attendance using attend_meeting(meeting_id, interval='SLOT_NUMBER').",
                    "CRITICAL: Look at the MEETING-SPECIFIC COMMON AVAILABILITY section above.",
                    "Find the row for your meeting. Use the SMALLEST slot number where value = 1.",
                    "Example: If Meeting 1 row shows Slot 2=1, Slot 3=1, Slot 7=1, use interval='2' (NOT '3' or '7').",
                    "NEVER use interval='skip' if any common slots (value=1) exist in the meeting row.",
                    "",
                ]
            )

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            context_parts.extend(
                [
                    "=== TOOL CALLING FORMAT ===",
                    phase_instructions,
                    "",
                ]
            )

        return "\n".join(context_parts)
