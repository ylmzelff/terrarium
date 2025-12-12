# pyright: basic
"""
PersonalAssistant Environment Adaptor

Adaptor to integrate the PersonalAssistant domain (outfit coordination)
with the black_boards_v5 communication protocol framework.

The PersonalAssistant environment involves agents coordinating to select outfits
that satisfy personal preferences and inter-agent constraints (color matching).
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple

# CoLLAB v2 problem-layer imports (made available via envs.dcops.__init__)
from problem_layer.personal_assistant import PersonalAssistantConfig, generate_instance
from problem_layer.personal_assistant.problem import Outfit
from problem_layer.base import ProblemDefinition
import logging
# Import abstract environment interface and logger
from envs.abstract_environment import AbstractEnvironment
from envs.dcops.plotter import ScorePlotter
from src.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
    build_plots_dir,
)
from .personal_assistant_tools import PersonalAssistantTools
from .personal_assistant_prompts import PersonalAssistantPrompts

# Use TYPE_CHECKING to avoid circular import (Agent → ToolsetDiscovery → EnvironmentTools → Environment → Agent)
if TYPE_CHECKING:
    from src.agent import Agent


class PersonalAssistantEnvironment(AbstractEnvironment):
    """
    PersonalAssistant environment adaptor for outfit coordination tasks.

    Agents coordinate to select outfits that satisfy personal preferences
    and inter-agent constraints (color matching/avoiding).
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize the PersonalAssistant environment."""
        self.full_config = config
        self.config: Dict[str, Any] = config["environment"]
        # Get the correct seed from environment config
        self.current_seed = self.config.get("rng_seed", 42)

        # Instance management
        # Partial joint assignment: variable_name -> chosen outfit number (1-based)
        self.assignment: Dict[str, Any] = {}
        self.outfit_selections: Dict[str, Outfit] = {}
        self.tool_logger = tool_logger
        self.agent_names: List[str] = []  # Agent names (renamed from agents_list)
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.current_iteration = 0
        self.simulation_config = config["simulation"]
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)

        # Clear seed directories FIRST to ensure clean state for this run
        clear_seed_directories("PersonalAssistant", self.current_seed, self.full_config)

        # ---- Build CoLLAB v2 instance -------------------------------------------------
        num_agents = self.config.get("num_agents", self.config.get("n_agents", 3))
        min_outfits = self.config.get("min_outfits_per_agent", 4)
        max_outfits = self.config.get("max_outfits_per_agent", 6)
        density = self.config.get("density")
        if density is None:
            # Back-compat: approximate density from max_degree if present
            max_degree = self.config.get("max_degree", 3)
            try:
                density = float(max_degree) / max(1.0, float(num_agents) - 1.0)
            except Exception:
                density = 0.3
        density = max(0.0, min(1.0, float(density)))

        collab_cfg = PersonalAssistantConfig(
            num_agents=int(num_agents),
            density=float(density),
            min_outfits_per_agent=int(min_outfits),
            max_outfits_per_agent=int(max_outfits),
            rng_seed=int(self.current_seed),
        )

        dcops_root = Path(__file__).resolve().parents[1]
        instance_dir = (
            dcops_root
            / "outputs"
            / "collab_instances"
            / "personal_assistant"
            / f"seed_{self.current_seed}"
        )
        self.instance = generate_instance(collab_cfg, instance_dir)
        self.problem: ProblemDefinition = self.instance.problem

        # Score tracking
        self.global_score_history: List[float] = []
        self.local_scores_history: Dict[str, List[float]] = {}
        self.score_plotter: Optional[ScorePlotter] = None
        self.agent_names = list(self.problem.agents.keys())
        self.agents: List['Agent'] = [] # Set this later in main.py in case agents get different clients or settings
        self.max_possible_score = float(getattr(self.instance, "max_utility", 0.0))

        # Initialize prompts (Put this after all other instance variables)
        # Note: tools are now in MCP server, not in environment
        self.prompts = PersonalAssistantPrompts(self, self.full_config)

        # Initialize score tracking
        self.local_scores_history = {agent: [] for agent in self.agent_names}

        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        plots_dir = build_plots_dir("PersonalAssistant", tag_model, self.current_seed, self.run_timestamp)
        self.score_plotter = ScorePlotter(save_dir=str(plots_dir))

        print(f"PersonalAssistant environment initialized with {len(self.agent_names)} agents")
        print(f"Agents: {', '.join(self.agent_names)}")
        print("PersonalAssistantEnvironment initialized")

    async def async_init(self):
        """Async initialization - create communication blackboards."""
        await self.create_comm_network()

    def set_agent_clients(self, agents: List['Agent']):
        """Set the agents for the environment."""
        self.agents = agents

    async def create_comm_network(self):
        """Create communication blackboards for multi‑agent coordination factors."""
        for factor in self.problem.factors:
            owners = {self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables}
            if len(owners) < 2:
                logging.warning(f"Skipping blackboard creation for meeting {meeting.meeting_id} with less than 2 participants")
                continue

            context = factor.description or "Coordination required between agents."
            blackboard_id = await self.communication_protocol.generate_comm_network(list(owners), context)
            print(f"Created Personal Assistant Blackboard {blackboard_id}: {list(owners)}")

    # NOTE: Changed from get_agents
    def get_agent_names(self) -> List[str]:
        """Get list of agent names."""
        return self.agent_names.copy()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int,
                          blackboard_contexts: Optional[Dict[str, str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn. This is used in the planning and execution phases
        in CommunicationProtocol.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            blackboard_contexts: Blackboard contexts from the protocol
            **kwargs: Additional context

        Returns:
            Dictionary with agent context
        """
        # Clear outfit selections at the start of each new iteration's planning phase
        # to allow agents to make new choices
        if phase == "planning" and iteration > 1 and (self.outfit_selections or self.assignment):
            print(f"PersonalAssistant: Clearing selections for iteration {iteration}")
            self.outfit_selections = {}
            self.assignment = {}

        if not self.instance:
            return {"error": "Environment not initialized"}

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "outfit_selections": self.outfit_selections.copy(),
            "total_agents": len(self.agent_names),
            "selections_made": len(self.outfit_selections),
            "selections_remaining": len(self.agent_names) - len(self.outfit_selections)
        }

        # Add configuration info (consistent with trading environment)
        assert self.config is not None, "Config not available"
        context["max_iterations"] = self.config.get("max_iterations", 10)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            context[key] = value

        return context

    def done(self, iteration: int) -> bool:
        """Return True when the environment is finished."""
        # Check max iterations first (consistent with trading environment)
        assert self.config is not None, "Config not available"
        max_iterations = self.config.get("max_iterations", 10)
        if iteration > max_iterations:
            print(f"Reached max iterations ({max_iterations}) - stopping simulation")
            return True

        # Stop early if all variables assigned and max utility reached
        total_vars = len(self.problem.variables)
        if len(self.assignment) == total_vars and self.instance:
            global_score, _ = self._compute_scores()
            if self.max_possible_score and global_score >= self.max_possible_score:
                print(
                    f"All constraints satisfied (score: {global_score}/{self.max_possible_score}) - simulation complete"
                )
                return True
            print(
                f"All agents selected but constraints not fully satisfied (score: {global_score}/{self.max_possible_score}) - continuing"
            )

        return False

    def _compute_scores(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute partial global and local utility for current assignment.

        Factors whose full scope has been assigned contribute to the total.
        Local utilities are attributed evenly to variable owners in scope.
        """
        total_utility = 0.0
        local_utilities: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}

        for factor in self.problem.factors:
            if not all(var in self.assignment for var in factor.scope):
                continue
            try:
                utility = factor.evaluate(self.assignment)
            except Exception:
                continue
            total_utility += utility

            owners = {self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables}
            if owners:
                share = utility / len(owners)
                for owner in owners:
                    local_utilities[owner] += share

        return total_utility, local_utilities

    def log_state(self, iteration: int, phase: str) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
            phase: Current phase
        """
        print(f"=== PersonalAssistant State - Iteration {iteration}, Phase {phase} ===")
        print(f"Agents: {len(self.agent_names)} total, {len(self.outfit_selections)} selected outfits")

        if self.outfit_selections:
            print("Current Selections:")
            for agent, outfit in self.outfit_selections.items():
                print(f"  {agent}: {outfit.article}, {outfit.color}")

        remaining = [agent for agent in self.agent_names if agent not in self.outfit_selections]
        if remaining:
            print(f"Remaining agents: {', '.join(remaining)}")

        global_score, local_scores = self._compute_scores()
        ratio = global_score / self.max_possible_score if self.max_possible_score else 0.0
        print(f"Current Global Score: {global_score:.2f} (ratio {ratio:.2%})")

        # Track scores and generate plots for every iteration
        self._track_scores(iteration, global_score, local_scores)

    def _track_scores(self, iteration: int, global_score: float, local_scores: Dict[str, float]) -> None:
        """Track scores and generate plots/logs."""
        import json
        from datetime import datetime

        # Update score histories
        self.global_score_history.append(global_score)
        for agent, score in local_scores.items():
            if agent in self.local_scores_history:
                self.local_scores_history[agent].append(score)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir("PersonalAssistant", tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": "PersonalAssistant",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "global_score": global_score/self.max_possible_score if self.max_possible_score > 0 else 0.0,
            "raw_global_score": global_score,
            "max_possible_score": self.max_possible_score,
            "local_scores": local_scores,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "metadata": {
                "total_agents": len(local_scores),
                "total_outfits_selected": len(self.outfit_selections),
                "average_local_score": sum(local_scores.values()) / len(local_scores) if local_scores else 0
            }
        }

        score_file = log_dir / f"scores_iteration_{iteration}.json"
        with open(score_file, 'w') as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

        # Generate plot
        if self.score_plotter:
            try:
                plot_path = self.score_plotter.plot_scores(
                    self.global_score_history,
                    self.local_scores_history,
                    iteration,
                    environment_name="PersonalAssistant",
                    show=False
                )
            except Exception as e:
                print(f"Warning: Failed to generate score plot: {e}")

    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Extract serializable state for MCP transmission.

        Returns:
            Dictionary with serializable environment state
        """
        # Extract wardrobe options in serializable format
        wardrobe_options: Dict[str, List[Dict[str, str]]] = {}
        if self.instance and getattr(self.instance, "wardrobe", None):
            for agent_name, outfits in self.instance.wardrobe.items():
                wardrobe_options[agent_name] = [
                    {"article": outfit.article, "color": outfit.color}
                    for outfit in outfits
                ]

        # Extract factors in serializable format
        factors: List[Dict[str, Any]] = []
        for factor in self.problem.factors:
            owners = sorted({self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables})
            factors.append({"name": factor.name, "type": factor.factor_type, "owners": owners})

        return {
            "outfit_selections": {
                agent: {"article": outfit.article, "color": outfit.color}
                for agent, outfit in self.outfit_selections.items()
            },
            "agent_names": self.agent_names.copy(),
            "wardrobe_options": wardrobe_options,
            "factors": factors,
            "max_possible_score": self.max_possible_score,
            "assignment": self.assignment.copy(),
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Args:
            state_updates: Dictionary with state updates to apply
        """
        if "outfit_selections" in state_updates:
            for agent, outfit_dict in state_updates["outfit_selections"].items():
                outfit = Outfit(
                    article=outfit_dict["article"],
                    color=outfit_dict["color"],
                    image=None,
                )
                self.outfit_selections[agent] = outfit

                # Update assignment using wardrobe index (1-based)
                options = self.instance.wardrobe.get(agent, []) if self.instance else []
                choice_num = None
                for idx, opt in enumerate(options, start=1):
                    if opt.article == outfit.article and opt.color == outfit.color:
                        choice_num = idx
                        break
                if choice_num is not None:
                    try:
                        var_name = self.problem.agent_variables(agent)[0].name
                        self.assignment[var_name] = choice_num
                    except Exception:
                        pass

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Post-tool execution callback for PersonalAssistant-specific processing.

        This is called after state updates are applied to perform environment-specific
        operations like score calculation.

        Args:
            state_updates: Dictionary with state updates that were applied
            response: The response dictionary to potentially modify
        """
        if "outfit_selections" in state_updates:
            global_score, _ = self._compute_scores()
            if "result" in response:
                response["result"]["global_score"] = global_score
                response["result"]["max_possible_score"] = self.max_possible_score

    def _generate_final_summary(self):
        """Generate final simulation summary."""
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE - FINAL SUMMARY")
        print("=" * 60)

        # Get environment-specific final summary
        final_summary = self.get_final_summary()

        print(f"Total iterations: {self.current_iteration}")

        if final_summary:
            for key, value in final_summary.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")

    def _log_iteration_summary(self, iteration: int):
        """Log the summary of an iteration."""
        print(f"\n--- ITERATION {iteration} SUMMARY ---")
        self.log_state(iteration, "iteration_end")

        # Get environment-specific summary
        summary = self.get_iteration_summary()
        if summary:
            print(f"  Iteration {iteration} summary:")
            for key, value in summary.items():
                print(f"    {key}: {value}")

    def cleanup(self) -> None:
        """Clean up any resources used by the environment."""
        print("PersonalAssistant environment cleanup")
        if self.outfit_selections:
            print(f"Final selections: {len(self.outfit_selections)}/{len(self.agent_names)} agents")

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current iteration for logging."""
        return {
            "selections_made": len(self.outfit_selections),
            "total_agents": len(self.agent_names),
            "completion_rate": len(self.outfit_selections) / len(self.agent_names) if self.agent_names else 0
        }

    def get_final_summary(self) -> Dict[str, Any]:
        """Get a final summary of the entire simulation."""
        total_vars = len(self.problem.variables)
        if not self.instance or len(self.assignment) != total_vars:
            return {
                "status": "incomplete",
                "variables_assigned": len(self.assignment),
                "total_variables": total_vars,
                "total_agents": len(self.agent_names),
            }

        global_score, local_scores = self._compute_scores()

        return {
            "status": "complete",
            "global_score": global_score / self.max_possible_score if self.max_possible_score > 0 else 0.0,
            "raw_global_score": global_score,
            "average_local_score": sum(local_scores.values()) / len(local_scores) if local_scores else 0.0,
            "local_scores": local_scores,
            "outfit_selections": {
                agent: {"article": outfit.article, "color": outfit.color}
                for agent, outfit in self.outfit_selections.items()
            }
        }
