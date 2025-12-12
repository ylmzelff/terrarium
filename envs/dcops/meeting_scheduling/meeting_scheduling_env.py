"""
MeetingScheduling Environment Adaptor of MeetingScheduling Environment in CoLLAB

Adaptor to integrate the MeetingScheduling domain.

The MeetingScheduling environment involves agents coordinating to decide
attendance intervals for a set of meetings on a shared timeline, following
the updated CoLLAB benchmark.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Mapping
import logging
# CoLLAB v2 problem-layer imports (made available via envs.dcops.__init__)
from problem_layer.meeting_scheduling import MeetingSchedulingConfig, generate_instance
from problem_layer.base import ProblemDefinition

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular import (Agent → ToolsetDiscovery → MeetingSchedulingEnvironmentTools → MeetingSchedulingEnvironment → Agent)
if TYPE_CHECKING:
    from src.agent import Agent
# Import abstract environment interface and loggers
from envs.abstract_environment import AbstractEnvironment
from src.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
)
from .meeting_scheduling_prompts import MeetingSchedulingPrompts

class MeetingSchedulingEnvironment(AbstractEnvironment):
    """
    MeetingScheduling environment adaptor for attendance‑interval coordination tasks.

    Agents decide how long to attend each meeting they are assigned to, aiming
    to maximize joint reward while avoiding overlaps.
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize the MeetingScheduling environment."""
        self.full_config = config
        self.config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        # Get the correct seed from environment config (matches what's used for instance generation)
        self.current_seed = self.config.get("rng_seed", 42)

        # Instance management
        # Partial joint assignment: variable_name -> chosen interval (e.g., "3-5" or "skip")
        self.assignment: Dict[str, Any] = {}
        self.tool_logger = tool_logger
        self.agent_names: List[str] = []
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self # Add bidirectional reference
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.current_iteration = 0
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)        
        # Clear seed directories FIRST to ensure clean state for this run
        clear_seed_directories("MeetingScheduling", self.current_seed, self.full_config)

        # ---- Build CoLLAB v2 instance -------------------------------------------------
        num_agents = self.config.get("num_agents", self.config.get("n_agents", 8))
        num_meetings = self.config.get("num_meetings", self.config.get("n_meetings", 6))
        timeline_length = self.config.get("timeline_length", 12)
        min_participants = self.config.get("min_participants", 2)
        max_participants = self.config.get(
            "max_participants", self.config.get("max_attendees_per_meeting", 4)
        )
        soft_ratio = self.config.get("soft_meeting_ratio", 0.6)

        collab_cfg = MeetingSchedulingConfig(
            num_agents=int(num_agents),
            num_meetings=int(num_meetings),
            timeline_length=int(timeline_length),
            min_participants=int(min_participants),
            max_participants=int(max_participants),
            soft_meeting_ratio=float(soft_ratio),
            rng_seed=int(self.current_seed),
        )

        dcops_root = Path(__file__).resolve().parents[1]
        instance_dir = (
            dcops_root
            / "outputs"
            / "collab_instances"
            / "meeting_scheduling"
            / f"seed_{self.current_seed}"
        )
        self.instance = generate_instance(collab_cfg, instance_dir)
        self.problem: ProblemDefinition = self.instance.problem

        # Score tracking
        self.joint_reward_history: List[float] = []
        self.agent_names = list(self.problem.agents.keys())
        self.max_joint_reward = float(getattr(self.instance, "max_utility", 0.0))
        self.agents: List['Agent'] = []

        # Initialize prompts (Put this after all other instance variables)
        self.prompts = MeetingSchedulingPrompts(self, self.full_config)

        # Initialize score tracking
        self.agent_rewards_history: Dict[str, List[float]] = {agent: [] for agent in self.agent_names}

        logger.info("MeetingScheduling environment initialized with %s agents", len(self.agent_names))
        logger.info("Agents: %s", ", ".join(self.agent_names))
        logger.info("Total meetings: %s", len(self.instance.meetings))
        logger.info("MeetingSchedulingEnvironment initialized")

    async def async_init(self):
        await self.create_comm_network()

    def set_agent_clients(self, agents: List['Agent']):
        """Set the agents for the environment."""
        self.agents = agents

    async def create_comm_network(self):
        """Create communication blackboards for each multi‑agent meeting."""
        # Create one blackboard per meeting (all participants coordinate there)
        for meeting in self.instance.meetings:
            participants = list(meeting.participants)
            if len(participants) < 2:
                logger.warning(
                    "Skipping blackboard creation for meeting %s with less than 2 participants",
                    meeting.meeting_id,
                )
                continue

            context = (
                f"Meeting {meeting.meeting_id}: {meeting.title} ({meeting.meeting_type}) "
                f"window [{meeting.start}, {meeting.end}). Participants: {', '.join(participants)}. "
                "Coordinate attendance intervals to maximise overlaps for soft meetings, "
                "fully attend strict meetings, and avoid double‑booking."
            )
            blackboard_id = await self.communication_protocol.generate_comm_network(participants, context)
            logger.info(
                "Created Meeting Blackboard %s: %s for %s",
                blackboard_id,
                participants,
                meeting.meeting_id,
            )

    def generate_final_summary(self):
        """Generate final simulation summary."""
        logger.info("%s", "=" * 60)
        logger.info("SIMULATION COMPLETE - FINAL SUMMARY")
        logger.info("%s", "=" * 60)

        # Get environment-specific final summary
        final_summary = self.get_final_summary()

        logger.info("Total iterations: %s", self.current_iteration)

        if final_summary:
            for key, value in final_summary.items():
                if isinstance(value, dict):
                    logger.info("%s:", key)
                    for sub_key, sub_value in value.items():
                        logger.info("  %s: %s", sub_key, sub_value)
                else:
                    logger.info("%s: %s", key, value)

    def log_iteration_summary(self, iteration: int):
        """Log the summary of an iteration."""
        logger.info("--- ITERATION %s SUMMARY ---", iteration)
        self.log_iteration(iteration)

    def get_agent_names(self) -> List[str]:
        """Get list of agent names."""
        return self.agent_names.copy()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            **kwargs: Additional context

        Returns:
            Dictionary with agent context
        """
        agent_vars = [var.name for var in self.problem.agent_variables(agent_name)]
        agent_choices = {v: self.assignment[v] for v in agent_vars if v in self.assignment}

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "joint_assignment": self.assignment.copy(),
            "agent_variables": agent_vars,
            "agent_choices": agent_choices,
            "timeline_length": self.instance.timeline_length,
            "total_variables": len(self.problem.variables),
            "variables_assigned": len(self.assignment),
            "variables_remaining": len(self.problem.variables) - len(self.assignment),
        }

        # Add configuration info
        context["max_iterations"] = self.config.get("max_iterations", 10)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            context[key] = value

        return context

    def done(self, iteration: int) -> bool:
        """Return True when the environment is finished."""
        # Check max iterations first
        assert self.config is not None, "Config not available"
        max_iterations = self.config.get("max_iterations", 10)
        if iteration > max_iterations:
            logger.info("Reached max iterations (%s) - stopping simulation", max_iterations)
            return True

        # Stop early if all variables have been assigned
        total_vars = len(self.problem.variables)
        if len(self.assignment) == total_vars:
            joint_reward = self.joint_reward(self.assignment)
            logger.info(
                "All attendance decisions made with joint reward: %.2f - simulation complete",
                joint_reward,
            )
            return True

        return False

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """Return the joint reward for a joint assignment."""
        total_reward, _ = self.rewards(actions)
        return total_reward

    def agent_reward(self, actions: Mapping[str, Any], agent: str) -> float:
        """Return the reward attributed to a single agent."""
        _, local_rewards = self.rewards(actions)
        assert agent in local_rewards, f"Agent {agent} not found in local rewards"
        local_reward = local_rewards.get(agent)
        assert local_reward is not None, f"Local reward for agent {agent} is None"
        return local_reward

    def rewards(self, actions: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute joint reward and per-agent rewards for a given joint assignment.

        We sum factor rewards whose full scope has been assigned.
        Per-agent rewards are attributed evenly to owners of variables in a factor's scope.
        """
        total_reward = 0.0
        local_rewards: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}

        for factor in self.problem.factors:
            if not all(var in actions for var in factor.scope):
                continue
            try:
                reward = factor.evaluate(actions)
            except Exception:
                continue
            total_reward += reward

            owners = {self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables}
            if owners:
                share = reward / len(owners)
                for owner in owners:
                    local_rewards[owner] += share

        return total_reward, local_rewards


    def log_iteration(self, iteration: int) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
        """
        logger.info("=== MeetingScheduling State - Iteration %s ===", iteration)
        total_vars = len(self.problem.variables)
        logger.info("Variables: %s total, %s assigned", total_vars, len(self.assignment))

        if self.assignment:
            logger.info("Current Attendance Decisions:")
            for var_name, value in sorted(self.assignment.items()):
                logger.info("  %s: %s", var_name, value)

        joint_reward, agent_rewards = self.rewards(self.assignment)
        ratio = joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0
        logger.info("Current Joint Reward: %.2f (ratio %.2f%%)", joint_reward, ratio * 100.0)

        # Track scores for every iteration
        self._track_scores(iteration, joint_reward, agent_rewards)

    def _track_scores(self, iteration: int, joint_reward: float, agent_rewards: Dict[str, float]) -> None:
        """Track scores and write logs."""
        import json
        from datetime import datetime

        # Update score histories
        self.joint_reward_history.append(joint_reward)
        for agent, reward in agent_rewards.items():
            if agent in self.agent_rewards_history:
                self.agent_rewards_history[agent].append(reward)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir("MeetingScheduling", tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": "MeetingScheduling",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0,
            "agent_rewards": agent_rewards,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "metadata": {
                "total_agents": len(agent_rewards),
                "variables_assigned": len(self.assignment),
                "total_variables": len(self.problem.variables),
                "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0.0,
            },
        }

        score_file = log_dir / f"scores_iteration_{iteration}.json"
        with open(score_file, 'w') as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Extract serializable state for MCP transmission.

        Returns:
            Dictionary with serializable environment state
        """
        meetings: Dict[str, Any] = {}
        for meeting in self.instance.meetings:
            meetings[meeting.meeting_id] = {
                "title": meeting.title,
                "meeting_type": meeting.meeting_type,
                "start": meeting.start,
                "end": meeting.end,
                "participants": list(meeting.participants),
            }

        return {
            "meetings": meetings,
            "attendance": self.assignment.copy(),
            "agent_names": self.agent_names.copy(),
            "timeline_length": self.instance.timeline_length,
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Args:
            state_updates: Dictionary with state updates to apply
        """
        # Apply attendance updates (UPDATE, don't replace!)
        if "attendance" in state_updates:
            self.assignment.update(state_updates["attendance"])

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Post-tool execution callback for MeetingScheduling-specific processing.
        Useful to get immediate score feedback, but not required.

        This is called after state updates are applied to perform environment-specific
        operations like score calculation.

        Args:
            state_updates: Dictionary with state updates that were applied
            response: The response dictionary to potentially modify
        """
        if "attendance" in state_updates:
            joint_reward = self.joint_reward(self.assignment)
            if "result" in response:
                response["result"]["joint_reward"] = joint_reward

    def get_final_summary(self) -> Dict[str, Any]:
        """Get a final summary of the entire simulation."""
        total_vars = len(self.problem.variables)
        final_attendance_decisions = f"{len(self.assignment)}/{total_vars} variables"
        if not self.instance or len(self.assignment) != total_vars:
            return {
                "status": "incomplete",
                "variables_assigned": len(self.assignment),
                "total_variables": total_vars,
                "total_agents": len(self.agent_names),
                "final_attendance_decisions": final_attendance_decisions,
            }

        joint_reward, agent_rewards = self.rewards(self.assignment)
        return {
            "status": "complete",
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0.0,
            "agent_rewards": agent_rewards,
            "attendance": self.assignment.copy(),
            "total_variables": total_vars,
            "variables_assigned": len(self.assignment),
            "total_agents": len(self.agent_names),
            "final_attendance_decisions": final_attendance_decisions,
        }
