import random
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass 
class SimpleProblem:
    """Minimal problem definition for Meeting Scheduling"""
    meetings: Dict[str, Any]
    agents: Dict[str, Any]  # Changed from List[str] to Dict to match interface
    description: str = "A simple meeting scheduling problem" # Added description field
    variables: Dict[str, Any] = None
    factors: List[Any] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
        if self.factors is None:
            self.factors = []
        # Ensure agents is a dict if passed as list (for backward compat/safety)
        if isinstance(self.agents, list):
            self.agents = {agent: {} for agent in self.agents}


    def agent_variables(self, agent_name):
        return []

@dataclass
class SimpleInstance:
    """Minimal instance definition"""
    problem: SimpleProblem
    timeline_length: int
    meetings: List[Any]
    max_utility: float = 100.0

@dataclass
class SimpleMeeting:
    """Minimal meeting structure"""
    meeting_id: str
    title: str
    start: int
    end: int
    meeting_type: str
    participants: List[str]

class MeetingSchedulingConfig:
    """Configuration class compatible with problem_layer"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# --- Personal Assistant Support ---

@dataclass
class Outfit:
    article: str
    color: str
    image: Any = None

class PersonalAssistantConfig(MeetingSchedulingConfig):
    pass

# --- Smart Grid Support ---

class SmartGridConfig(MeetingSchedulingConfig):
    pass

# --- Expanded Generator ---

def generate_instance(config, instance_dir=None):
    """
    Generates a simple random instance when the real problem_layer is missing.
    Supports MeetingScheduling, PersonalAssistant, and SmartGrid via config type or attributes.
    """
    # Detect config type
    is_smart_grid = hasattr(config, 'min_machines_per_agent') or isinstance(config, SmartGridConfig)
    is_personal_assistant = hasattr(config, 'min_outfits_per_agent') or isinstance(config, PersonalAssistantConfig)
    
    if is_smart_grid:
        return _generate_smart_grid_instance(config)
    elif is_personal_assistant:
        return _generate_personal_assistant_instance(config)
    else:
        return _generate_meeting_instance(config)

def _generate_meeting_instance(config):
    # Extract config parameters with defaults
    num_meetings = getattr(config, 'num_meetings', 4)
    num_agents = getattr(config, 'num_agents', 5) 
    timeline_length = getattr(config, 'timeline_length', 12)
    min_participants = getattr(config, 'min_participants', 2)
    max_participants = getattr(config, 'max_participants', 4)
    
    meetings_dict = {}
    meetings_list = []
    agent_names = [f"Agent_{i}" for i in range(num_agents)]
    
    for i in range(num_meetings):
        meeting_id = f"m{i+1:03d}"
        # Random start/duration
        start_time = random.randint(0, timeline_length-3)
        duration = random.randint(2, min(4, timeline_length - start_time))
        end_time = start_time + duration
        
        # Random participants
        k = random.randint(min_participants, min(max_participants, num_agents))
        participants = random.sample(agent_names, k)
        
        # Create meeting object
        meeting_obj = SimpleMeeting(
            meeting_id=meeting_id,
            title=f"Meeting {i+1}",
            start=start_time,
            end=end_time,
            meeting_type="soft" if random.random() > 0.5 else "strict",
            participants=participants
        )
        
        meetings_list.append(meeting_obj)
        
        meetings_dict[meeting_id] = {
            "id": meeting_id,
            "title": f"Meeting {i+1}",
            "start": start_time,
            "end": end_time, 
            "meeting_type": meeting_obj.meeting_type,
            "participants": participants
        }
    
    problem = SimpleProblem(meetings=meetings_dict, agents=agent_names)
    return SimpleInstance(
        problem=problem,
        timeline_length=timeline_length,
        meetings=meetings_list
    )

def _generate_smart_grid_instance(config):
    num_agents = getattr(config, 'num_agents', 5)
    
    # Mock data
    problem = SimpleProblem(meetings={}, agents=[f"Agent_{i}" for i in range(num_agents)])
    instance = SimpleInstance(
        problem=problem,
        timeline_length=24,
        meetings=[]
    )
    # Add dummy attrs expected by SmartGridEnv
    instance.machines = {}
    instance.sources = {}
    instance.machine_powers = {}
    instance.min_utility = 0.0
    return instance

def _generate_personal_assistant_instance(config):
    num_agents = getattr(config, 'num_agents', 5)
    
    # Mock data
    problem = SimpleProblem(meetings={}, agents=[f"Agent_{i}" for i in range(num_agents)])
    instance = SimpleInstance(
        problem=problem, 
        timeline_length=1, 
        meetings=[]
    )
    # Add dummy attrs expected by PersonalAssistantEnv
    instance.wardrobe = {}
    return instance

