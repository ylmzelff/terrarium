"""MeetingScheduling Environment

Agents coordinate to schedule meetings at earliest common available slots
using privacy-preserving Oblivious Transfer (OT) protocol.
"""
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Mapping

import os
import logging
logger = logging.getLogger(__name__)

# Azure credentials are read EXCLUSIVELY from examples/configs/meeting_scheduling.yaml
# (environment.graph_api block).  No env var fallbacks — YAML is the single source of truth.

# Use TYPE_CHECKING to avoid circular import (BaseAgent → ToolsetDiscovery → MeetingSchedulingEnvironmentTools → MeetingSchedulingEnvironment → BaseAgent)
if TYPE_CHECKING:
    from src.agents.base import BaseAgent
# Import abstract environment interface and loggers
from envs.abstract_environment import AbstractEnvironment
from src.utils import (
    clear_seed_directories,
    get_run_timestamp,
)
from .meeting_scheduling_prompts import MeetingSchedulingPrompts

class SimpleMeeting:
    """Simple meeting object."""
    def __init__(self, meeting_id: str, title: str, participants: List[str]):
        self.meeting_id = meeting_id
        self.title = title
        self.participants = participants


class SimpleProblemDefinition:
    """Problem definition for meeting scheduling."""
    def __init__(self, agents: List[str], meetings: List[SimpleMeeting]):
        self.agents = {name: name for name in agents}
        self.variables = []
        self._agent_vars = {agent: [] for agent in agents}
        self.description = (
            f"Meeting scheduling task with {len(agents)} agents and {len(meetings)} meeting(s). "
            "Coordinate to schedule meetings at earliest common available slots."
        )
        
        # Create variables (one per agent per meeting they participate in)
        for meeting in meetings:
            for agent in meeting.participants:
                var_name = f"{agent}__{meeting.meeting_id}"
                self.variables.append(var_name)
                self._agent_vars[agent].append(var_name)
    
    def agent_variables(self, agent_name: str):
        """Return variables (meeting assignments) for an agent."""
        class Variable:
            def __init__(self, name):
                self.name = name
        return [Variable(v) for v in self._agent_vars.get(agent_name, [])]
    
    def agent_instruction(self, agent_name: str) -> str:
        """Return instruction for agent (simplified)."""
        num_meetings = len(self._agent_vars.get(agent_name, []))
        return f"You participate in {num_meetings} meeting(s). Select the earliest available slot from the intersection for each meeting."


class MeetingSchedulingEnvironment(AbstractEnvironment):
    """MeetingScheduling environment for multi-agent coordination.

    Agents coordinate attendance decisions while respecting availability
    constraints discovered via OT protocol.
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize the MeetingScheduling environment."""
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        # Get the correct seed from simulation config (matches what's used for instance generation)
        self.current_seed = int(self.simulation_config["seed"])

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
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        # Generate agent names
        network_cfg = config.get("communication_network") or {}
        assert network_cfg is not None and network_cfg != {}, "communication_network config must be specified"
        num_agents = network_cfg.get("num_agents")
        assert num_agents is not None and type(num_agents) == int, "communication_network.num_agents must be an integer"

        num_meetings = self.env_config.get("num_meetings", 1)
        
        # Generate agent names
        agent_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.agent_names = [f"Agent{agent_letters[i]}" for i in range(num_agents)]
        
        # Generate simple meetings
        self.meetings = []
        for i in range(num_meetings):
            meeting_id = f"m{i+1:03d}"
            title = f"Meeting {i+1}"
            # All agents participate in all meetings for simplicity
            participants = self.agent_names.copy()
            self.meetings.append(SimpleMeeting(meeting_id, title, participants))
        
        # Create simple problem definition
        class SimpleInstance:
            def __init__(self, meetings):
                self.meetings = meetings
        
        self.instance = SimpleInstance(self.meetings)
        self.problem = SimpleProblemDefinition(self.agent_names, self.meetings)

        self.agents: List["BaseAgent"] = []

        # Initialize prompts (Put this after all other instance variables)
        self.prompts = MeetingSchedulingPrompts(self, self.full_config)

        # Availability table configuration (from config or defaults)
        self.num_days = self.env_config.get("num_days", 1)
        self.slots_per_day = self.env_config.get("slots_per_day", 24)

        # ── Agentic mode: LLM agents fetch their own calendars via tools ──
        # These are populated at runtime when agents call submit_availability_array.
        self.meeting_availabilities: Dict[str, Dict[str, List[int]]] = {}
        # Tracks which agents have submitted arrays per meeting:
        # { meeting_id: { agent_name: [0,1,...] } }
        self._submitted_arrays: Dict[str, Dict[str, List[int]]] = {}

        # Cache for meeting intersections (computed once via OT, reused everywhere)
        self._meeting_intersections_cache: Optional[Dict] = None

        # Pre-generate and cache simulated availability (deterministic via seed)
        # Used when use_real_calendars=False so tools return consistent data
        self._simulated_availability_cache: Dict[str, Dict[str, List[int]]] = {}
        if not self.env_config.get("use_real_calendars", False):
            for meeting in self.meetings:
                self._simulated_availability_cache[meeting.meeting_id] = (
                    self._generate_simulated_availability(meeting.participants)
                )

        logger.info("%s initialized with %s agents", self.__class__.__name__, len(self.agent_names))
        logger.info("Agent Names: %s", ", ".join(self.agent_names))
        logger.info("Total meetings to schedule: %s", len(self.meetings))
        logger.info("Agentic mode: LLM agents will fetch their own calendars via tools.")
        logger.info("Slot config: %d days × %d slots/day = %d total",
                    self.num_days, self.slots_per_day, self.num_days * self.slots_per_day)

    async def async_init(self):
        await super().async_init()

        use_real = self.env_config.get("use_real_calendars", False)

        if use_real:
            # ── HARD VALIDATION: Graph API MUST work or simulation dies ──
            logger.info("=" * 80)
            logger.info("🔒 use_real_calendars=true → Validating Graph API connection at startup")
            logger.info("=" * 80)

            # 1. Check dependencies
            try:
                import msal, requests, pytz  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "❌ SIMULATION CANNOT START: Graph API dependencies missing.\n"
                    "   Run: pip install msal requests pytz\n"
                    f"   Error: {exc}"
                ) from exc

            # 2. Check YAML config has required fields
            graph_config = self.env_config.get("graph_api", {})
            client_id = graph_config.get("client_id")
            tenant_id = graph_config.get("tenant_id")

            if not client_id:
                raise RuntimeError(
                    "❌ SIMULATION CANNOT START: graph_api.client_id is missing in YAML config.\n"
                    "   Add it under environment.graph_api.client_id in your YAML file."
                )
            if not tenant_id:
                raise RuntimeError(
                    "❌ SIMULATION CANNOT START: graph_api.tenant_id is missing in YAML config.\n"
                    "   Add it under environment.graph_api.tenant_id in your YAML file."
                )

            # 3. Check agent_emails mapping
            agent_emails = graph_config.get("agent_emails", {})
            for agent_name in self.agent_names:
                if agent_name not in agent_emails or not agent_emails[agent_name]:
                    raise RuntimeError(
                        f"❌ SIMULATION CANNOT START: No email configured for agent '{agent_name}'.\n"
                        f"   Add it under environment.graph_api.agent_emails.{agent_name} in YAML."
                    )

            # 4. Initialize Graph API client and authenticate
            try:
                from llm_server.clients.graph_client import GraphAPIClient
                self._graph_client = GraphAPIClient.from_yaml(self.full_config)
                logger.info("✅ Graph API client initialized from YAML config.")
            except Exception as exc:
                raise RuntimeError(
                    f"❌ SIMULATION CANNOT START: Graph API client initialization failed.\n"
                    f"   Error: {exc}\n"
                    f"   Check your YAML config: client_id={client_id}, tenant_id={tenant_id}"
                ) from exc

            # 5. Authenticate for each agent (device code flow)
            for agent_name in self.agent_names:
                email = agent_emails[agent_name]
                try:
                    logger.info("🔑 Authenticating %s (%s)...", agent_name, email)
                    token = self._graph_client.get_token_for_user(email=email)
                    if not token:
                        raise RuntimeError(f"Authentication returned empty token for {email}")
                    logger.info("✅ %s (%s) authenticated successfully.", agent_name, email)
                except Exception as exc:
                    raise RuntimeError(
                        f"❌ SIMULATION CANNOT START: Authentication failed for {agent_name} ({email}).\n"
                        f"   Error: {exc}\n"
                        f"   Make sure you complete the device code flow for this account."
                    ) from exc

            logger.info("=" * 80)
            logger.info("✅ ALL Graph API VALIDATIONS PASSED — real calendars will be used.")
            logger.info("=" * 80)
        else:
            logger.info("🔬 Simulation mode — using pre-generated availability data.")

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
            "total_variables": len(self.problem.variables),
            "variables_assigned": len(self.assignment),
            "variables_remaining": len(self.problem.variables) - len(self.assignment),
        }

        # Add configuration info
        context["max_iterations"] = self.env_config.get("max_iterations", 1)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            context[key] = value

        return context

    def done(self, iteration: int) -> bool:
        """Return True when the environment is finished."""
        # Check max iterations first
        assert self.env_config is not None, "Config not available"
        max_iterations = self.env_config.get("max_iterations", 1)
        if iteration > max_iterations:
            logger.info("Reached max iterations (%s) - stopping simulation", max_iterations)
            return True

        # Stop early if all variables have been assigned
        total_vars = len(self.problem.variables)
        if len(self.assignment) == total_vars:
            logger.info("All attendance decisions made - simulation complete")
            return True
        return False
    
    def compute_max_joint_reward(self) -> float:
        """Rewarding is disabled for this environment."""
        return 0.0

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """Rewarding is disabled for this environment."""
        return 0.0

    def agent_reward(self, actions: Mapping[str, Any], agent: str) -> float:
        """Rewarding is disabled for this environment."""
        return 0.0


    def _generate_availability_for_meeting(self, participants: List[str]) -> Dict[str, List[int]]:
        """
        Generate availability arrays for meeting participants.
        
        This is the main entry point for availability generation. It dispatches to either:
        - Simulation mode (_generate_simulated_availability): Controlled test data
        - Production mode (_fetch_real_availability): Real Outlook calendars via Graph API
        
        Mode is controlled by 'use_real_calendars' config parameter.
        
        Args:
            participants: List of agent names participating in this meeting
            
        Returns:
            Dictionary mapping agent names to their availability lists for this meeting
        """
        use_real_calendars = self.env_config.get("use_real_calendars", False)
        
        if use_real_calendars:
            logger.info("📅 Production mode: Fetching real availability from Microsoft Graph API")
            return self._fetch_real_availability(participants)
        else:
            logger.info("🔬 Simulation mode: Generating controlled test availability")
            return self._generate_simulated_availability(participants)
    
    def _generate_simulated_availability(self, participants: List[str]) -> Dict[str, List[int]]:
        """
        Generate SIMULATED availability arrays for testing.
        
        Generation logic mirrors examples/generate_availability_configs.py:
        - Each agent gets approximately availability_rate fraction of available slots.
        - A guaranteed number of COMMON slots are injected for all participants.
        - Non-intersection available slots are sampled without overlap where possible,
          so pairwise overlap is primarily controlled by the guaranteed intersections.
        
        Args:
            participants: List of agent names participating in this meeting
            
        Returns:
            Dictionary mapping agent names to their availability lists for this meeting
        """
        import random
        from src.availability import AvailabilityConstants
        
        # Get configuration parameters
        total_slots = self.num_days * self.slots_per_day
        availability_rate = float(self.env_config.get('availability_rate', 0.4))  # 40% default
        availability_rate = max(0.0, min(1.0, availability_rate))

        # Supports two config styles:
        # - intersection or intersections: absolute slot count (preferred for this env)
        # - intersection_density: fraction of total slots (optional compatibility)
        intersections_cfg = self.env_config.get('intersections') or self.env_config.get('intersection', 0)
        intersection_density_cfg = float(self.env_config.get('intersection_density', 0.0))
        intersection_density_cfg = max(0.0, min(1.0, intersection_density_cfg))

        num_available_per_agent = round(total_slots * availability_rate)

        if isinstance(intersections_cfg, float) and 0.0 <= intersections_cfg <= 1.0:
            requested_intersections = round(total_slots * intersections_cfg)
        else:
            requested_intersections = int(intersections_cfg) if intersections_cfg else 0
        if requested_intersections <= 0 and intersection_density_cfg > 0.0:
            requested_intersections = round(total_slots * intersection_density_cfg)

        requested_intersections = max(0, requested_intersections)
        num_intersections = min(requested_intersections, num_available_per_agent, total_slots)
        
        logger.debug(f"Intersection config: intersections_cfg={intersections_cfg}, "
                    f"requested={requested_intersections}, "
                    f"available_per_agent={num_available_per_agent}, "
                    f"final_num_intersections={num_intersections}")

        base_rng = random.Random(self.current_seed)
        if num_intersections > 0:
            intersection_slots = sorted(base_rng.sample(range(total_slots), num_intersections))
        else:
            intersection_slots = []
        
        logger.info("=" * 80)
        logger.info("📊 GENERATING AVAILABILITY ARRAYS FOR SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Total slots: {total_slots}")
        logger.info(f"Availability rate target: ~{int(availability_rate*100)}%")
        logger.info(f"Guaranteed intersections target: {num_intersections}")
        logger.info("-" * 80)
        
        availability = {}
        used_non_intersection_slots = set()
        
        # Generate arrays with guaranteed common slots and controlled extra slots
        for agent_idx, agent_name in enumerate(participants):
            # Use a stable per-agent seed for reproducible but distinct arrays.
            agent_seed = self.current_seed + ((agent_idx + 1) * 1009)
            rng = random.Random(agent_seed)
            
            slots = [AvailabilityConstants.BUSY] * total_slots  # Start with all busy

            # Step 1: force guaranteed common slots for all participants
            for slot_idx in intersection_slots:
                slots[slot_idx] = AvailabilityConstants.AVAILABLE

            # Step 2: fill additional available slots for this agent
            num_additional = max(0, num_available_per_agent - num_intersections)
            remaining_slots = [i for i in range(total_slots) if i not in intersection_slots]
            remaining_slots = [i for i in remaining_slots if i not in used_non_intersection_slots]

            actual_additional = min(num_additional, len(remaining_slots))
            if actual_additional > 0:
                additional_slots = rng.sample(remaining_slots, actual_additional)
                for slot_idx in additional_slots:
                    slots[slot_idx] = AvailabilityConstants.AVAILABLE
                used_non_intersection_slots.update(additional_slots)
            else:
                additional_slots = []

            if actual_additional < num_additional:
                logger.warning(
                    "Agent %s requested %d additional slots but only %d were available "
                    "after enforcing non-overlap.",
                    agent_name,
                    num_additional,
                    actual_additional,
                )
            
            availability[agent_name] = slots
            
            # Log agent availability summary
            available_count = sum(slots)
            available_indices = [i for i, val in enumerate(slots) if val == 1]
            logger.info(f"\n🔷 {agent_name}:")
            logger.info(f"   Available slots: {available_count}/{total_slots} ({available_count/total_slots*100:.1f}%)")
            logger.info(f"   Guaranteed common slots (shared): {len(intersection_slots)}")
            logger.info(f"   Additional private slots: {len(additional_slots)}")
            logger.info(f"   Available indices: {available_indices}")
            logger.info(f"   Full array: {slots}")
        
        logger.info("-" * 80)
        logger.info(f"✅ Guaranteed intersection slots: {intersection_slots}")
        logger.info("ℹ️  OT protocol will compute/verify the actual common slots from these arrays")
        logger.info("=" * 80)
        
        return availability

    def _fetch_real_availability(self, participants: List[str]) -> Dict[str, List[int]]:
        """
        Fetch REAL availability from Microsoft Outlook calendars via Graph API.
        
        FAILS HARD: If anything goes wrong (missing email, API error, etc.),
        a RuntimeError is raised and the simulation stops. No silent fallbacks.
        
        Args:
            participants: List of agent names participating in this meeting
            
        Returns:
            Dictionary mapping agent names to their availability lists
            
        Raises:
            RuntimeError: If ANY agent's calendar cannot be fetched
        """
        from datetime import datetime, timedelta
        from src.availability import AvailabilityConstants
        
        try:
            from llm_server.clients.graph_client import GraphAPIClient
        except ImportError as exc:
            raise RuntimeError(
                "❌ SIMULATION STOPPED: Graph API dependencies missing.\n"
                "   Run: pip install msal requests pytz"
            ) from exc
        
        # Graph client should already be initialized in async_init
        if not hasattr(self, '_graph_client'):
            try:
                self._graph_client = GraphAPIClient.from_yaml(self.full_config)
            except Exception as e:
                raise RuntimeError(
                    f"❌ SIMULATION STOPPED: Failed to initialize Graph API client: {e}"
                ) from e
        
        # Get agent email mapping
        graph_config = self.env_config.get("graph_api", {})
        agent_emails = graph_config.get("agent_emails", {})
        
        # Calculate time window
        start_datetime = datetime.now()
        end_datetime = start_datetime + timedelta(days=self.num_days)
        
        availability = {}
        total_slots = self.num_days * self.slots_per_day
        
        for agent_name in participants:
            email = agent_emails.get(agent_name)
            
            if not email:
                raise RuntimeError(
                    f"❌ SIMULATION STOPPED: No email configured for agent '{agent_name}'.\n"
                    f"   Add it under environment.graph_api.agent_emails.{agent_name} in YAML."
                )
            
            # Fetch availability — NO try/except fallback, let it crash
            logger.info(f"📅 Fetching availability for {agent_name} ({email})...")
            
            raw_availability = self._graph_client.get_availability(
                email=email,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                interval_minutes=60
            )
            
            # Convert Graph API response to our slot format
            slots = self._convert_graph_availability_to_slots(
                raw_availability,
                total_slots
            )
            
            availability[agent_name] = slots
            
            available_count = sum(slots)
            logger.info(f"  ✅ {agent_name}: {available_count}/{total_slots} available slots")
        
        return availability
    
    def _convert_graph_availability_to_slots(
        self,
        raw_availability: List[Dict],
        total_slots: int
    ) -> List[int]:
        """
        Convert Graph API availability response to binary slot array.
        
        Graph API returns:
        [
            {"start": "2026-02-21T09:00:00", "end": "2026-02-21T10:00:00", "status": "free"},
            {"start": "2026-02-21T10:00:00", "end": "2026-02-21T11:00:00", "status": "busy"},
            ...
        ]
        
        We convert to: [1, 0, ...] where 1=available, 0=busy
        
        Args:
            raw_availability: List of time slots with status from Graph API
            total_slots: Expected total number of slots
            
        Returns:
            Binary availability array
        """
        from src.availability import AvailabilityConstants
        
        slots = []
        
        for slot_data in raw_availability[:total_slots]:
            status = slot_data.get("status", "busy")
            
            # Map Graph API status to our binary format
            # "free" = available (1), everything else = busy (0)
            is_available = status == "free"
            
            slots.append(
                AvailabilityConstants.AVAILABLE if is_available
                else AvailabilityConstants.BUSY
            )
        
        # Pad with busy slots if Graph API returned fewer slots than expected
        while len(slots) < total_slots:
            slots.append(AvailabilityConstants.BUSY)
        
        return slots[:total_slots]  # Trim if too many

    # ─────────────────────────────────────────────────────────────────────
    # Agentic Tools — called by MeetingSchedulingTools handlers
    # ─────────────────────────────────────────────────────────────────────

    def fetch_calendar_for_agent(self, agent_name: str, meeting_id: str) -> Dict[str, Any]:
        """
        Called by the fetch_my_calendar tool.

        Returns the agent's calendar events so they can build their availability array.
        - If use_real_calendars=False: Returns pre-generated simulated availability
        - If use_real_calendars=True: Fetches real Outlook/Teams calendar via Graph API

        Returns a dict the LLM receives as tool output:
          {
            "events":      [ {"slot_index": ..., "start": ..., "end": ..., "status": "free"|"busy"}, ... ],
            "slot_info":   { "total_slots": 120, "slot_duration_minutes": 60, ... },
            "instructions": "..."
          }
        """
        from datetime import datetime, timedelta
        
        use_real_calendars = self.env_config.get("use_real_calendars", False)
        total_slots = self.num_days * self.slots_per_day
        
        if use_real_calendars:
            # ════════════════════════════════════════════════════════════════
            # REAL CALENDARS MODE: Fetch from Microsoft Graph API
            # ════════════════════════════════════════════════════════════════
            graph_config = self.env_config.get("graph_api", {})
            agent_emails = graph_config.get("agent_emails", {})
            email = agent_emails.get(agent_name)

            if not email:
                raise RuntimeError(
                    f"❌ SIMULATION STOPPED: No email configured for agent '{agent_name}'.\n"
                    f"   Add it under environment.graph_api.agent_emails.{agent_name} in YAML."
                )

            # Lazy-init Graph API client (should already exist from async_init)
            if not hasattr(self, "_graph_client"):
                from llm_server.clients.graph_client import GraphAPIClient
                self._graph_client = GraphAPIClient.from_yaml(self.full_config)
                logger.info("✅ Graph API client initialised from YAML config for agentic tool flow.")

            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt   = start_dt + timedelta(days=self.num_days)

            logger.info("📅 [%s] Fetching REAL calendar from Graph API (%s → %s)", 
                        agent_name, start_dt.date(), end_dt.date())
            
            raw_slots = self._graph_client.get_availability(
                email=email,
                start_datetime=start_dt,
                end_datetime=end_dt,
                interval_minutes=60,
            )

            # Convert to event list
            events = [
                {
                    "slot_index": i,
                    "start":  s["start"],
                    "end":    s["end"],
                    "status": s["status"],   # "free" | "busy"
                }
                for i, s in enumerate(raw_slots)
            ]

            busy_count = sum(1 for e in events if e["status"] == "busy")
            logger.info("📋 [%s] Returned %d REAL calendar slots (%d busy, %d free) for meeting %s",
                        agent_name, len(events), busy_count, len(events) - busy_count, meeting_id)
        
        else:
            # ════════════════════════════════════════════════════════════════
            # SIMULATION MODE: Return pre-generated random availability
            # ════════════════════════════════════════════════════════════════
            logger.info("🔬 [%s] Returning SIMULATED calendar for meeting %s", 
                        agent_name, meeting_id)
            
            # Get simulated availability from cache (generated during __init__)
            if meeting_id not in self._simulated_availability_cache:
                raise RuntimeError(
                    f"❌ No simulated availability cached for meeting {meeting_id}. "
                    f"This should have been generated during __init__."
                )
            
            agent_availability = self._simulated_availability_cache[meeting_id].get(agent_name)
            if agent_availability is None:
                raise RuntimeError(
                    f"❌ No simulated availability for agent {agent_name} in meeting {meeting_id}"
                )
            
            # Convert binary array to event list format (same as Graph API)
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            events = []
            for slot_idx, is_available in enumerate(agent_availability):
                slot_start = start_dt + timedelta(hours=slot_idx)
                slot_end = slot_start + timedelta(hours=1)
                
                events.append({
                    "slot_index": slot_idx,
                    "start": slot_start.isoformat(),
                    "end": slot_end.isoformat(),
                    "status": "free" if is_available == 1 else "busy",
                })
            
            busy_count = sum(1 for e in events if e["status"] == "busy")
            logger.info("📋 [%s] Returned %d SIMULATED slots (%d busy, %d free) for meeting %s",
                        agent_name, len(events), busy_count, len(events) - busy_count, meeting_id)

        # ════════════════════════════════════════════════════════════════
        # Return same format regardless of mode
        # ════════════════════════════════════════════════════════════════
        return {
            "meeting_id": meeting_id,
            "agent":      agent_name,
            "events":     events,
            "slot_info": {
                "total_slots":           total_slots,
                "slots_per_day":         self.slots_per_day,
                "num_days":              self.num_days,
                "slot_duration_minutes": 60,
                "work_hours":            "09:00–18:00 (slots outside this range are always busy)",
            },
            "instructions": (
                "Build a binary availability array of length "
                f"{total_slots} (= {self.num_days} days × {self.slots_per_day} slots/day). "
                "Rules:\n"
                "  • 1 = you are FREE at that slot\n"
                "  • 0 = you are BUSY at that slot\n"
                "  • Slots outside 09:00–18:00 work hours → always 0\n"
                "  • Slots where status='busy' → 0\n"
                "  • Slots where status='free' AND within work hours → 1\n"
                "After building the array, call submit_availability_array("
                f"meeting_id='{meeting_id}', availability=[...]) with your array."
            ),
        }

    def submit_availability_array(
        self,
        agent_name: str,
        meeting_id: str,
        availability: List[int],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Called by the submit_availability_array tool.

        Stores the agent's binary availability.  When ALL participants of the meeting
        have submitted their arrays the OT protocol is triggered automatically and the
        intersection is written to every relevant blackboard.
        """
        from src.availability import AvailabilityConstants

        total_slots = self.num_days * self.slots_per_day

        # ── Validate ──────────────────────────────────────────────────────
        if len(availability) != total_slots:
            return {
                "status": "error",
                "reason": (
                    f"Expected {total_slots} values ({self.num_days} days × {self.slots_per_day} slots/day), "
                    f"got {len(availability)}."
                ),
            }

        # ── Store the submitted array ──────────────────────────────────────
        if meeting_id not in self._submitted_arrays:
            self._submitted_arrays[meeting_id] = {}
        self._submitted_arrays[meeting_id][agent_name] = availability

        free_count = sum(availability)
        logger.info(
            "📥 Availability submitted | meeting=%s | agent=%s | free=%d/%d",
            meeting_id, agent_name, free_count, total_slots,
        )

        # Find the meeting's participants
        meeting_obj  = next((m for m in self.meetings if m.meeting_id == meeting_id), None)
        participants = meeting_obj.participants if meeting_obj else list(self._submitted_arrays[meeting_id].keys())
        submitted    = self._submitted_arrays[meeting_id]
        waiting_for  = [p for p in participants if p not in submitted]

        if waiting_for:
            return {
                "status":      "received",
                "meeting_id":  meeting_id,
                "agent":       agent_name,
                "free_slots":  free_count,
                "waiting_for": waiting_for,
                "message":     f"Array saved. Waiting for {waiting_for} to also submit.",
            }

        # ── All participants submitted → run OT ───────────────────────────
        logger.info("🔒 All participants submitted → running OT for meeting %s", meeting_id)

        # Populate meeting_availabilities so existing OT/blackboard code works unchanged
        self.meeting_availabilities[meeting_id] = {
            p: submitted[p] for p in participants
        }
        # Invalidate intersection cache so OT re-runs with new data
        self._meeting_intersections_cache = None

        try:
            intersections = self._generate_meeting_intersections()
        except Exception as exc:
            logger.exception("OT protocol failed for meeting %s: %s", meeting_id, exc)
            return {"status": "error", "reason": f"OT protocol failed: {exc}"}

        intersection = intersections.get(meeting_id, [])
        common_indices = [i for i, v in enumerate(intersection) if v == AvailabilityConstants.AVAILABLE]

        # Write intersection to all relevant blackboards
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._async_log_availability_table())
            else:
                loop.run_until_complete(self._async_log_availability_table())
        except Exception as exc:
            logger.warning("Could not write availability table to blackboard: %s", exc)

        return {
            "status":         "ot_complete",
            "meeting_id":     meeting_id,
            "common_slots":   len(common_indices),
            "intersection":   intersection,
            "common_indices": common_indices,
            "message": (
                f"OT complete. {len(common_indices)} common slots found. "
                "The intersection has been posted to the shared blackboard. "
                "Coordinate with the other agent to select the EARLIEST common slot (smallest index)."
            ),
        }


    def _generate_availability_slots(self) -> Dict[str, List[int]]:
        """
        DEPRECATED: Use meeting_availabilities instead.
        This method aggregates availability across all meetings for backward compatibility.
        """
        # Return first meeting's availability if exists, otherwise empty
        if self.meeting_availabilities:
            first_meeting_id = list(self.meeting_availabilities.keys())[0]
            return self.meeting_availabilities[first_meeting_id]
        return {}

    def _generate_meeting_intersections(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate intersection (common availability) for each meeting using OT protocol.
        
        Uses privacy-preserving Oblivious Transfer (OT) protocol to compute intersection
        without revealing individual availability preferences. Requires exactly 2 participants.
        
        Results are cached after first computation to avoid re-running OT protocol.
        
        Returns:
            Dictionary mapping meeting IDs to intersection availability
            
        Raises:
            ValueError: If meeting doesn't have exactly 2 participants
            ImportError: If OT module is not available
        """
        from src.availability import AvailabilityConstants
        
        # Return cached result if available
        if self._meeting_intersections_cache is not None:
            logger.debug("Using cached meeting intersections (OT already computed)")
            return self._meeting_intersections_cache
        
        meeting_intersections = {}
        total_slots = self.num_days * self.slots_per_day
        
        for meeting_id, agent_slots in self.meeting_availabilities.items():
            participants = list(agent_slots.keys())
            
            if not participants:
                continue
            
            # Privacy-preserving OT protocol (REQUIRED - no fallback)
            if len(participants) != 2:
                raise ValueError(
                    f"Meeting {meeting_id} has {len(participants)} participants. "
                    f"OT protocol requires exactly 2 participants. "
                    f"Multi-party OT not yet implemented."
                )
            
            # Import OT module (fail fast if not available)
            try:
                from crypto import compute_private_intersection
            except ImportError as e:
                raise ImportError(
                    f"OT module not available. This is REQUIRED for privacy-preserving intersection.\n"
                    f"Install with: cd crypto && python setup.py install\n"
                    f"Original error: {e}"
                ) from e
            
            import time
            
            sender, receiver = participants[0], participants[1]
            # Use binary arrays directly (0/1 availability)
            sender_availability = agent_slots[sender]
            receiver_availability = agent_slots[receiver]
            
            # Extract indices for logging
            sender_indices = [i for i in range(total_slots) if sender_availability[i] == AvailabilityConstants.AVAILABLE]
            receiver_indices = [i for i in range(total_slots) if receiver_availability[i] == AvailabilityConstants.AVAILABLE]
            
            logger.info("=" * 80)
            logger.info(f"🔒 PRIVACY-PRESERVING OBLIVIOUS TRANSFER (OT) PROTOCOL")
            logger.info("=" * 80)
            logger.info(f"Meeting: {meeting_id}")
            logger.info(f"Protocol: Priority Oblivious Transfer (5-phase)")
            logger.info(f"Sender: {sender} ({len(sender_indices)} available slots)")
            logger.info(f"Receiver: {receiver} ({len(receiver_indices)} available slots)")
            logger.info(f"Total slots: {total_slots}")
            logger.info("-" * 80)
            logger.info(f"📥 OT INPUT ARRAYS:")
            logger.info(f"")
            logger.info(f"   {sender} (Sender):")
            logger.info(f"      Binary array: {sender_availability}")
            logger.info(f"      Available indices: {sender_indices}")
            logger.info(f"      Available count: {len(sender_indices)}/{total_slots}")
            logger.info(f"")
            logger.info(f"   {receiver} (Receiver):")
            logger.info(f"      Binary array: {receiver_availability}")
            logger.info(f"      Available indices: {receiver_indices}")
            logger.info(f"      Available count: {len(receiver_indices)}/{total_slots}")
            logger.info("-" * 80)
            logger.info("⚙️  Executing OT phases: Setup → GenQuery → GenRes → oblFilter → Retrieve")
            
            start_time = time.time()
            
            # OT returns intersection indices directly (NO fallback, NO classical AND)
            common_indices = compute_private_intersection(sender_availability, receiver_availability, total_slots)
            
            ot_duration = time.time() - start_time
            
            # Convert indices back to slot array
            intersection = [AvailabilityConstants.BUSY] * total_slots
            for idx in common_indices:
                intersection[idx] = AvailabilityConstants.AVAILABLE
            
            logger.info(f"✓ OT Protocol Complete (Duration: {ot_duration:.4f}s)")
            logger.info("-" * 80)
            logger.info(f"📊 OT OUTPUT - INTERSECTION RESULT:")
            logger.info(f"")
            logger.info(f"   Common slots found: {len(common_indices)}/{total_slots} ({len(common_indices)/total_slots*100:.1f}%)")
            logger.info(f"   Intersection indices: {common_indices}")
            logger.info(f"   Intersection binary array: {intersection}")
            logger.info(f"")
            logger.info(f"   🔐 Privacy guarantee: ✓ NO individual availability disclosed")
            logger.info(f"      • {sender} does NOT know {receiver}'s individual slots")
            logger.info(f"      • {receiver} does NOT know {sender}'s individual slots")
            logger.info(f"      • Both parties ONLY know: {common_indices} (intersection)")
            logger.info("=" * 80)
            
            meeting_intersections[meeting_id] = intersection
        
        # Cache the results for reuse
        self._meeting_intersections_cache = meeting_intersections
        logger.debug(f"Cached meeting intersections for {len(meeting_intersections)} meeting(s)")
        
        return meeting_intersections

    async def _async_log_availability_table(self) -> None:
        """
        Log agent availability table to all relevant blackboards during planning phase via MCP.
        
        Uses OT protocol to compute privacy-preserving intersections for each meeting,
        then logs both individual agent availability and meeting-specific intersection
        tables to relevant blackboards.
        
        Each meeting must have exactly 2 participants for OT protocol.
        """
        from src.availability import AvailabilityConstants
        
        try:
            logger.info("=" * 80)
            logger.info("📋 LOGGING AVAILABILITY TABLES TO BLACKBOARDS")
            logger.info("=" * 80)
            
            # Generate meeting intersections
            meeting_intersections = self._generate_meeting_intersections()
            
            # Prepare meeting info for formatting
            meeting_info = {}
            for meeting in self.meetings:
                meeting_info[meeting.meeting_id] = {
                    "title": meeting.title,
                    "participants": meeting.participants
                }
            
            # For each meeting, log its availability table
            for meeting_id, agent_slots in self.meeting_availabilities.items():
                meeting_data = meeting_info.get(meeting_id, {})
                meeting_title = meeting_data.get("title", meeting_id)
                participants = meeting_data.get("participants", [])
                
                # Get intersection for this meeting
                intersection = meeting_intersections.get(meeting_id, [])
                common_slots = [i for i, val in enumerate(intersection) if val == AvailabilityConstants.AVAILABLE]
                
                logger.info(f"Meeting {meeting_id} ({meeting_title}):")
                logger.info(f"  Participants: {', '.join(participants)}")
                logger.info(f"  Common slots: {len(common_slots)} → {common_slots}")
                
                # Get all blackboards via MCP
                async with self.communication_protocol.mcp_client as client:
                    blackboards_result = await client.call_tool("return_blackboards", {})
                    
                    # Parse MCP response (handle TextContent and other formats)
                    import json
                    blackboards = None
                    
                    # Direct list/dict
                    if isinstance(blackboards_result, list):
                        blackboards = blackboards_result
                    # Has content attribute (TextContent or similar)
                    elif hasattr(blackboards_result, 'content'):
                        content = blackboards_result.content
                        # TextContent with text attribute (JSON string)
                        if hasattr(content, 'text'):
                            try:
                                blackboards = json.loads(content.text)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse blackboards JSON: {content.text}")
                                blackboards = []
                        # Content is already a list
                        elif isinstance(content, list):
                            blackboards = content
                    # Has data attribute
                    elif hasattr(blackboards_result, 'data'):
                        blackboards = blackboards_result.data
                    # Fallback: try JSON parse
                    else:
                        try:
                            blackboards = json.loads(str(blackboards_result))
                        except:
                            blackboards = []
                
                if not blackboards:
                    logger.warning(f"No blackboards available for logging {meeting_title}")
                    continue
                
                # Log to each blackboard with relevant agents
                for blackboard in blackboards:
                    # Parse blackboard object (handle TextContent, dict, or object)
                    blackboard_id = None
                    blackboard_agents = None
                    
                    if isinstance(blackboard, dict):
                        # Direct dict
                        blackboard_id = int(blackboard['blackboard_id'])
                        blackboard_agents = blackboard['agents']
                    # TextContent with text attribute (JSON string)
                    elif hasattr(blackboard, 'text'):
                        try:
                            parsed = json.loads(blackboard.text)
                            # If parsed result is a list, take first element
                            if isinstance(parsed, list):
                                if parsed:
                                    blackboard_id = int(parsed[0]['blackboard_id'])
                                    blackboard_agents = parsed[0]['agents']
                            else:
                                blackboard_id = int(parsed['blackboard_id'])
                                blackboard_agents = parsed['agents']
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                            logger.error(f"Failed to parse blackboard JSON: {blackboard.text}, error: {e}")
                            continue
                    # Object with attributes
                    elif hasattr(blackboard, 'blackboard_id'):
                        blackboard_id = int(blackboard.blackboard_id)
                        blackboard_agents = blackboard.agents
                    else:
                        logger.warning(f"Unknown blackboard format: {type(blackboard)}")
                        continue
                    
                    # Skip if extraction failed
                    if blackboard_id is None or blackboard_agents is None:
                        logger.warning(f"Could not extract blackboard data from: {blackboard}")
                        continue
                    
                    # Filter agents that are participants in THIS meeting AND this blackboard
                    relevant_agents = [a for a in participants if a in blackboard_agents]
                    
                    if not relevant_agents:
                        continue
                    
                    # Create filtered agent slots for this blackboard (only this meeting's participants)
                    filtered_slots = {agent: agent_slots[agent] for agent in relevant_agents}
                    
                    # Prepare meeting-specific intersection
                    meeting_intersections_for_bb = {meeting_id: intersection}
                    meeting_info_for_bb = {meeting_id: meeting_data}
                    
                    # Log to blackboard via MCP
                    async with self.communication_protocol.mcp_client as client:
                        tool_result = await client.call_tool("log_availability_table", {
                            "blackboard_id": blackboard_id,
                            "agent_slots": filtered_slots,
                            "num_days": self.num_days,
                            "num_slots_per_day": self.slots_per_day,
                            "phase": AvailabilityConstants.PHASE_PLANNING,
                            "meeting_intersections": meeting_intersections_for_bb,
                            "meeting_info": meeting_info_for_bb
                        })
                        
                        # Parse result (handle TextContent and other formats)
                        result = None
                        
                        # Direct dict
                        if isinstance(tool_result, dict):
                            result = tool_result
                        # Has content attribute (TextContent or similar)
                        elif hasattr(tool_result, 'content'):
                            content = tool_result.content
                            # TextContent with text attribute (JSON string)
                            if hasattr(content, 'text'):
                                try:
                                    result = json.loads(content.text)
                                except json.JSONDecodeError:
                                    result = {"status": "error", "message": "Invalid JSON response"}
                            # Content is already a dict
                            elif isinstance(content, dict):
                                result = content
                            # Content is a list (might contain TextContent items)
                            elif isinstance(content, list):
                                if content and hasattr(content[0], 'text'):
                                    try:
                                        result = json.loads(content[0].text)
                                    except (json.JSONDecodeError, AttributeError):
                                        result = {"status": "error", "message": "Failed to parse list content"}
                                else:
                                    result = {"status": "error", "message": f"Unknown list content: {type(content[0]) if content else 'empty'}"}
                            else:
                                logger.debug(f"Unknown content type: {type(content)}, value: {content}")
                                result = {"status": "error", "message": f"Unknown content format: {type(content).__name__}"}
                        # Has data attribute
                        elif hasattr(tool_result, 'data'):
                            result = tool_result.data
                        # Fallback
                        else:
                            result = tool_result
                        
                        if isinstance(result, dict) and result.get("status") == "success":
                            logger.info(
                                f"  ✓ Blackboard {blackboard_id}: Logged to agents {', '.join(relevant_agents)}"
                            )

                            # Also post explicit OT input/output details so blackboard logs show
                            # exactly what entered OT and what intersection was produced.
                            if len(relevant_agents) == 2:
                                sender_name = relevant_agents[0]
                                receiver_name = relevant_agents[1]
                                sender_array = filtered_slots.get(sender_name, [])
                                receiver_array = filtered_slots.get(receiver_name, [])
                                sender_indices = [i for i, v in enumerate(sender_array) if v == AvailabilityConstants.AVAILABLE]
                                receiver_indices = [i for i, v in enumerate(receiver_array) if v == AvailabilityConstants.AVAILABLE]
                                intersection_indices = [i for i, v in enumerate(intersection) if v == AvailabilityConstants.AVAILABLE]

                                ot_payload = {
                                    "message": f"OT details for {meeting_id}: {len(intersection_indices)} common slot(s)",
                                    "meeting_id": meeting_id,
                                    "participants": relevant_agents,
                                    "sender": sender_name,
                                    "receiver": receiver_name,
                                    "sender_array": sender_array,
                                    "receiver_array": receiver_array,
                                    "sender_available_indices": sender_indices,
                                    "receiver_available_indices": receiver_indices,
                                    "intersection_array": intersection,
                                    "intersection_indices": intersection_indices,
                                    "common_slots": len(intersection_indices),
                                    "phase": AvailabilityConstants.PHASE_PLANNING,
                                    "iteration": int(self.current_iteration),
                                }

                                ot_result_raw = await client.call_tool("post_system_message", {
                                    "blackboard_id": blackboard_id,
                                    "kind": "ot_protocol",
                                    "payload": ot_payload,
                                })

                                ot_result = ot_result_raw
                                if hasattr(ot_result_raw, "content"):
                                    content = ot_result_raw.content
                                    if hasattr(content, "text"):
                                        try:
                                            ot_result = json.loads(content.text)
                                        except json.JSONDecodeError:
                                            ot_result = {"status": "error", "message": "Invalid JSON response"}

                                if isinstance(ot_result, dict) and ot_result.get("status") == "error":
                                    logger.warning(
                                        "  ✗ Blackboard %s: OT detail event failed - %s",
                                        blackboard_id,
                                        ot_result,
                                    )
                                else:
                                    logger.info(
                                        "  ✓ Blackboard %s: OT detail event posted (%d common slots)",
                                        blackboard_id,
                                        len(intersection_indices),
                                    )
                        else:
                            logger.warning(
                                f"  ✗ Blackboard {blackboard_id}: Failed - {result}"
                            )
            
            logger.info("=" * 80)
            logger.info(f"✓ AVAILABILITY TABLES LOGGED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ FAILED TO LOG AVAILABILITY TABLE")
            logger.error("=" * 80)
            logger.error(f"Error: {e}", exc_info=True)
            logger.warning("Simulation will continue without availability table")
            logger.error("=" * 80)


    def log_iteration(self, iteration: int) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
        """
        logger.info("=== %s State - Iteration %s ===", self.__class__.__name__, iteration)
        # Note: Availability table is now logged in async_init() before planning starts
        
        total_vars = len(self.problem.variables)
        logger.info("Variables: %s total, %s assigned", total_vars, len(self.assignment))

        if self.assignment:
            logger.info("Current Attendance Decisions:")
            for var_name, value in sorted(self.assignment.items()):
                logger.info("  %s: %s", var_name, value)

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

        return {
            "status": "complete",
            "attendance": self.assignment.copy(),
            "total_variables": total_vars,
            "variables_assigned": len(self.assignment),
            "total_agents": len(self.agent_names),
            "final_attendance_decisions": final_attendance_decisions,
        }

    #### MCP-specific methods ####

    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Extract serializable state for MCP transmission.

        Returns:
            Dictionary with serializable environment state
        """
        meetings: Dict[str, Any] = {}
        total_slots = self.num_days * self.slots_per_day
        for meeting in self.instance.meetings:
            meetings[meeting.meeting_id] = {
                "title":        meeting.title,
                "participants": list(meeting.participants),
                "start":        0,
                "end":          total_slots,
            }

        # Include graph_api config so tools can use it without a direct env reference
        graph_api = self.env_config.get("graph_api", {})

        # Include use_real_calendars so tools know whether to call Graph API or use simulation
        use_real_calendars = self.env_config.get("use_real_calendars", False)

        # Include simulated availability when in simulation mode so tools can return it
        # without needing Graph API
        simulated_availability = {}
        if not use_real_calendars:
            simulated_availability = self._simulated_availability_cache

        # Expose OT intersections (if already computed) so execution tool can enforce
        # scheduling at the earliest common slot.
        meeting_intersections = self._meeting_intersections_cache or {}
        earliest_common_slots: Dict[str, int] = {}
        for meeting_id, intersection in meeting_intersections.items():
            common_indices = [i for i, v in enumerate(intersection) if v == 1]
            if common_indices:
                earliest_common_slots[meeting_id] = common_indices[0]

        return {
            "meetings":     meetings,
            "attendance":   self.assignment.copy(),
            "agent_names":  self.agent_names.copy(),
            "num_days":     self.num_days,
            "slots_per_day": self.slots_per_day,
            "graph_api":    graph_api,
            "use_real_calendars": use_real_calendars,
            "simulated_availability": simulated_availability,
            "meeting_intersections": meeting_intersections,
            "earliest_common_slots": earliest_common_slots,
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Handles:
          - attendance: agent → meeting slot assignments (from attend_meeting)
          - submitted_arrays: agent binary availability arrays (from submit_availability_array)
            When all participants of a meeting have submitted, OT is triggered automatically.
        """
        logger.info(f"📥 apply_state_updates called with keys: {list(state_updates.keys())}")
        
        # ── Attendance updates (attend_meeting) ───────────────────────────
        if "attendance" in state_updates:
            self.assignment.update(state_updates["attendance"])

        # ── Submitted availability arrays (submit_availability_array) ─────
        if "submitted_arrays" in state_updates:
            incoming = state_updates["submitted_arrays"]
            logger.info(f"📨 Processing submitted_arrays: {list(incoming.keys())}")
            # incoming = { meeting_id: { agent_name: [0, 1, ...] } }
            for meeting_id, agent_arrays in incoming.items():
                if meeting_id not in self._submitted_arrays:
                    self._submitted_arrays[meeting_id] = {}
                self._submitted_arrays[meeting_id].update(agent_arrays)

                # Check if all participants have submitted
                meeting_obj  = next((m for m in self.meetings if m.meeting_id == meeting_id), None)
                participants = meeting_obj.participants if meeting_obj else list(agent_arrays.keys())
                submitted    = self._submitted_arrays[meeting_id]
                all_done     = all(p in submitted for p in participants)

                logger.info(f"   Meeting {meeting_id}: participants={participants}, submitted={list(submitted.keys())}, all_done={all_done}")
                
                if all_done:
                    logger.info("🔒 All arrays received for %s → running OT", meeting_id)
                    self.meeting_availabilities[meeting_id] = {p: submitted[p] for p in participants}
                    self._meeting_intersections_cache = None  # force OT re-run

                    try:
                        self._generate_meeting_intersections()
                        logger.info("✅ OT complete for %s → writing to blackboard", meeting_id)
                        # Schedule async blackboard write
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.ensure_future(self._async_log_availability_table())
                            else:
                                loop.run_until_complete(self._async_log_availability_table())
                        except Exception as exc:
                            logger.warning("Could not write availability table to blackboard: %s", exc)
                    except Exception as exc:
                        logger.error("❌ OT protocol failed for %s: %s", meeting_id, exc)

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
            return
