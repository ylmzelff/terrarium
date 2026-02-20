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
            execution_tool_lines=[
                "- attend_meeting(meeting_id: str, interval: str): "
                "Schedule your attendance at a specific time slot. "
                "interval should be the SLOT NUMBER where intersection = 1 (e.g., '2' for Slot 2, '3' for Slot 3). "
                "IMPORTANT: Check the MEETING-SPECIFIC COMMON AVAILABILITY table. "
                "Select the SMALLEST slot number where value = 1 in the meeting intersection row."
            ],
            planning_header="Planning phase tools (blackboard coordination only):",
            execution_header="Execution phase tools (blackboard + attendance decisions):",
            system_note=(
                "Planning: only blackboard tools are permitted.\n"
                "Execution: attend_meeting becomes available to commit to the earliest available slot."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = """You are participating in a meeting scheduling coordination task.

TASK:
- You have an availability array showing your free/busy time slots.
- Other agents have their own availability arrays.
- The system calculates the INTERSECTION (common available slots) between all participants.
- Your goal: Schedule the meeting at the EARLIEST common available slot.
  → Look for slots where intersection = 1 in the MEETING-SPECIFIC COMMON AVAILABILITY table
  → Select the SMALLEST slot number (e.g., Slot 2 comes before Slot 3)
  → Example: If "Meeting 1" row shows Slot 2=1, Slot 3=1, Slot 7=1, you MUST select interval='2'

PHASES:
- Planning Phase: Use blackboards to discuss and identify the earliest common available slot.
- Execution Phase: Commit your attendance at the agreed slot using the attend_meeting tool.

RULES:
- You may only schedule meetings you participate in.
- Check the MEETING-SPECIFIC COMMON AVAILABILITY section in the availability table.
- Find the row for your meeting. It shows where ALL participants are available (value = 1).
- Select the slot with the SMALLEST NUMBER where value = 1 in that meeting's row.
- Use attend_meeting(meeting_id, interval='SLOT_NUMBER') to commit.
- Example: If Meeting 1 row shows Slot 2=1, Slot 5=1, Slot 7=1, choose interval='2' (NOT '5' or '7')."""

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
                    "Use blackboards to identify the earliest common available slot.",
                    "Look at the MEETING-SPECIFIC COMMON AVAILABILITY table.",
                    "Find the SMALLEST slot number where intersection = 1.",
                    "Coordinate with other participants to agree on this slot.",
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
