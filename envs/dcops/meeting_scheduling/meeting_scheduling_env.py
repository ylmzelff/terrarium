"""MeetingScheduling Environment

Agents coordinate to schedule meetings at earliest common available slots
using privacy-preserving Oblivious Transfer (OT) protocol.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Mapping

import logging
logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular import (BaseAgent → ToolsetDiscovery → MeetingSchedulingEnvironmentTools → MeetingSchedulingEnvironment → BaseAgent)
if TYPE_CHECKING:
    from src.agents.base import BaseAgent
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

    Agents coordinate attendance decisions to maximize joint reward
    while respecting availability constraints discovered via OT protocol.
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

        # Score tracking
        self.joint_reward_history: List[float] = []
        self.max_joint_reward = float(num_meetings)  # 1.0 point per meeting scheduled
        self.agents: List["BaseAgent"] = []

        # Initialize prompts (Put this after all other instance variables)
        self.prompts = MeetingSchedulingPrompts(self, self.full_config)

        # Initialize score tracking
        self.agent_rewards_history: Dict[str, List[float]] = {agent: [] for agent in self.agent_names}

        # Availability table configuration (from config or defaults)
        self.num_days = self.env_config.get("num_days", 1)
        self.slots_per_day = self.env_config.get("slots_per_day", 12)
        
        # Generate availability for each meeting
        self.meeting_availabilities = {}  # meeting_id -> {agent_name -> [0,1,0,...]}
        for meeting in self.meetings:
            self.meeting_availabilities[meeting.meeting_id] = self._generate_availability_for_meeting(
                meeting.participants
            )
        
        # Cache for meeting intersections (computed once via OT, reused everywhere)
        self._meeting_intersections_cache = None

        logger.info("%s initialized with %s agents", self.__class__.__name__, len(self.agent_names))
        logger.info("Agent Names: %s", ", ".join(self.agent_names))
        logger.info("Total meetings to schedule: %s", len(self.meetings))
        logger.info("Availability table: %d days x %d slots/day", self.num_days, self.slots_per_day)

    async def async_init(self):
        await super().async_init()
        # Log availability table at initialization, BEFORE any planning rounds start
        # This ensures the table appears in agent prompts from the first round
        logger.info("🚀 Initializing availability tables...")
        await self._async_log_availability_table()

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
            joint_reward = self.joint_reward(self.assignment)
            logger.info(
                "All attendance decisions made with joint reward: %.2f - simulation complete",
                joint_reward,
            )
            return True
        return False
    
    def compute_max_joint_reward(self) -> float:
        """Return the optimal joint reward (1.0 per agent per successful meeting)."""
        # Maximum reward = all agents successfully scheduled all meetings at earliest common slot
        return float(len(self.agent_names) * len(self.meetings))

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """Return the joint reward for a joint assignment."""
        total_reward, _ = self._rewards(actions)
        return total_reward

    def agent_reward(self, actions: Mapping[str, Any], agent: str) -> float:
        """Return the reward attributed to a single agent."""
        _, local_rewards = self._rewards(actions)
        return local_rewards.get(agent, 0.0)

    def _rewards(self, actions: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute rewards based on successful slot selections.
        
        Reward = 1.0 for each agent that selected a valid slot from the intersection.
        """
        total_reward = 0.0
        local_rewards: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}
        
        # For each meeting, check if agents selected slots from the intersection
        meeting_intersections = self._generate_meeting_intersections()
        
        for meeting_id, intersection in meeting_intersections.items():
            # Find intersection slots (where value = 1)
            valid_slots = [i for i, val in enumerate(intersection) if val == 1]
            
            if not valid_slots:
                continue
            
            # Check each agent's choice for this meeting
            for agent in self.agent_names:
                var_name = f"{agent}__{meeting_id}"
                if var_name in actions:
                    chosen_interval = actions[var_name]
                    
                    # Parse interval (e.g., "3" or "3-4")
                    try:
                        if '-' in str(chosen_interval):
                            start = int(chosen_interval.split('-')[0])
                        else:
                            start = int(chosen_interval)
                        
                        # Reward if chosen slot is in valid intersection slots
                        if start in valid_slots:
                            total_reward += 1.0
                            local_rewards[agent] += 1.0
                    except (ValueError, AttributeError):
                        pass  # Invalid format, no reward
        
        return total_reward, local_rewards


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
        Generate SIMULATED availability arrays with GUARANTEED intersections.
        
        The 'intersections' config parameter controls how many slots ALL participants
        are guaranteed to be available. This creates a controlled test scenario where
        the OT protocol is guaranteed to find exactly N common slots.
        
        Args:
            participants: List of agent names participating in this meeting
            
        Returns:
            Dictionary mapping agent names to their availability lists for this meeting
        """
        import random
        from src.availability import AvailabilityConstants
        
        # Get configuration parameters
        total_slots = self.num_days * self.slots_per_day
        availability_rate = self.env_config.get('availability_rate', 0.4)  # 40% default
        num_intersections = self.env_config.get('intersections', 3)  # Default: 3 common slots
        
        # Validate intersections parameter
        if num_intersections > total_slots:
            logger.warning(
                f"intersections={num_intersections} > total_slots={total_slots}, "
                f"capping to {total_slots}"
            )
            num_intersections = total_slots
        
        logger.info(
            f"Generating controlled availability: {total_slots} total slots, "
            f"{num_intersections} GUARANTEED intersections (common slots)"
        )
        
        # Step 1: Select N random slots that will be the GUARANTEED intersection
        # These slots will be available (1) for ALL participants
        intersection_slots = random.sample(range(total_slots), num_intersections)
        intersection_slots.sort()  # Sort for readability
        
        logger.info(f"  🎯 Guaranteed intersection slots: {intersection_slots}")
        
        availability = {}
        
        # Step 2: Generate availability for each participant
        for agent_idx, agent_name in enumerate(participants):
            # Use agent-specific random seed for reproducibility but agent-level variation
            # This ensures different agents have different availability patterns
            agent_seed = self.current_seed + hash(agent_name) % 10000
            rng = random.Random(agent_seed)
            
            slots = [AvailabilityConstants.BUSY] * total_slots  # Start with all busy
            
            # Step 2a: Set intersection slots to AVAILABLE for this agent
            for slot_idx in intersection_slots:
                slots[slot_idx] = AvailabilityConstants.AVAILABLE
            
            # Step 2b: Randomly set other slots based on availability_rate
            # (but never override the guaranteed intersection slots)
            remaining_slots = [i for i in range(total_slots) if i not in intersection_slots]
            for slot_idx in remaining_slots:
                is_available = rng.random() < availability_rate  # Agent-specific random
                slots[slot_idx] = (
                    AvailabilityConstants.AVAILABLE if is_available 
                    else AvailabilityConstants.BUSY
                )
            
            availability[agent_name] = slots
            
            # Log agent availability summary
            available_count = sum(slots)
            available_indices = [i for i, val in enumerate(slots) if val == 1]
            logger.info(
                f"  {agent_name}: {available_count}/{total_slots} available slots"
            )
            logger.debug(f"    Available indices: {available_indices}")
        
        logger.info(
            f"  ✅ Guaranteed: ALL {len(participants)} participants available at slots {intersection_slots}"
        )
        
        return availability

    def _fetch_real_availability(self, participants: List[str]) -> Dict[str, List[int]]:
        """
        Fetch REAL availability from Microsoft Outlook calendars via Graph API.
        
        This requires:
        - Azure App Registration with proper permissions
        - Environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
        - User authentication credentials
        - graph_api config section with agent_emails mapping
        
        Args:
            participants: List of agent names participating in this meeting
            
        Returns:
            Dictionary mapping agent names to their availability lists for this meeting
            
        Raises:
            ImportError: If graph_client dependencies not installed
            Exception: If Graph API call fails
        """
        from datetime import datetime, timedelta
        from src.availability import AvailabilityConstants
        
        try:
            from llm_server.clients.graph_client import GraphAPIClient
        except ImportError:
            logger.error(
                "❌ Graph API client not available. Install dependencies: "
                "pip install msal requests pytz"
            )
            raise
        
        # Initialize Graph API client (lazy loading, reuse if already created)
        if not hasattr(self, '_graph_client'):
            graph_config = self.env_config.get("graph_api", {})
            
            timezone = graph_config.get("timezone", "UTC")
            
            try:
                # Create client from environment variables
                self._graph_client = GraphAPIClient.from_env(timezone=timezone)
                logger.info("✅ Graph API client initialized - Device Flow Authentication will trigger if tokens are missing.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Graph API client: {e}")
                raise
        
        # Get agent email mapping
        graph_config = self.env_config.get("graph_api", {})
        agent_emails = graph_config.get("agent_emails", {})
        
        # Calculate time window (next num_days days)
        start_datetime = datetime.now()
        end_datetime = start_datetime + timedelta(days=self.num_days)
        
        availability = {}
        total_slots = self.num_days * self.slots_per_day
        
        for agent_name in participants:
            email = agent_emails.get(agent_name)
            
            if not email:
                logger.error(f"❌ No email configured for agent: {agent_name}")
                # Fallback to empty availability
                availability[agent_name] = [AvailabilityConstants.BUSY] * total_slots
                continue
            
            try:
                # Fetch availability from Graph API
                logger.info(f"📅 Fetching availability for {agent_name} ({email})...")
                
                # Get schedule in 1-hour slots (Graph API limitation)
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
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch availability for {agent_name}: {e}")
                # Fallback to empty availability
                availability[agent_name] = [AvailabilityConstants.BUSY] * total_slots
        
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
            logger.info(f"Input (Sender {sender} binary):   {sender_availability}")
            logger.info(f"Input (Receiver {receiver} binary): {receiver_availability}")
            logger.info(f"Sender available indices:          {sender_indices}")
            logger.info(f"Receiver available indices:        {receiver_indices}")
            logger.info("-" * 80)
            logger.info("Executing OT phases: Setup → GenQuery → GenRes → oblFilter → Retrieve")
            
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
            logger.info(f"📊 RESULTS:")
            logger.info(f"   Common slots found: {len(common_indices)}/{total_slots}")
            logger.info(f"   Intersection indices: {common_indices}")
            logger.info(f"   Privacy guarantee: ✓ NO individual availability disclosed")
            logger.info(f"   {sender} does NOT know {receiver}'s individual slots")
            logger.info(f"   {receiver} does NOT know {sender}'s individual slots")
            logger.info(f"   Both parties ONLY know: {common_indices} (intersection)")
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

        joint_reward, agent_rewards = self._rewards(self.assignment)
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
        log_dir = build_log_dir(self.__class__.__name__, tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": self.__class__.__name__,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward,
            "max_joint_reward": self.max_joint_reward,
            "agent_rewards": agent_rewards,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards),
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "total_agents": len(agent_rewards),
            "variables_assigned": len(self.assignment),
            "total_variables": len(self.problem.variables),
        }

        data_file = log_dir / f"data_iteration_{iteration}.json"
        with open(data_file, "w") as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

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

        joint_reward, agent_rewards = self._rewards(self.assignment)
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
                "title": meeting.title,
                "participants": list(meeting.participants),
                "start": 0,
                "end": total_slots,
                "meeting_type": "soft",
            }

        return {
            "meetings": meetings,
            "attendance": self.assignment.copy(),
            "agent_names": self.agent_names.copy(),
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
