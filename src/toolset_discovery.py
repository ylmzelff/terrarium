from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time
import uuid
from envs.dcops.meeting_scheduling.meeting_scheduling_tools import MeetingSchedulingTools
from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools
from envs.dcops.smart_grid.smartgrid_tools import SmartGridTools

class ToolsetDiscovery:
    def __init__(self):
        self.meeting_tools = MeetingSchedulingTools(blackboard_manager=None)
        self.personal_assistant_tools = PersonalAssistantTools(blackboard_manager=None)
        self.smartgrid_tools = SmartGridTools(blackboard_manager=None)

    def get_tools_for_environment(self, environment_name: str, phase: str) -> List[Dict[str, Any]]:
        """
        Get the toolset for a specific environment.

        Args:
            environment_name: Name of the environment (normalized to lowercase, e.g., "meetingscheduling")
            phase: Current phase ("planning" or "execution")
        """
        # Normalize environment name for comparison (remove underscores and lowercase)
        normalized = environment_name.lower().replace("_", "")

        if normalized == "meetingscheduling":
            return self.meeting_tools.get_tools(phase)
        elif normalized == "personalassistant":
            return self.personal_assistant_tools.get_tools(phase)
        elif normalized == "smartgrid":
            return self.smartgrid_tools.get_tools(phase)
        else:
            return []

    def get_env_tool_names(self, environment_name: str) -> Set[str]:
        """
        Get the set of tool names that this environment supports.

        Args:
            environment_name: Name of the environment (normalized to lowercase, e.g., "meetingscheduling")
        """
        # Normalize environment name for comparison (remove underscores and lowercase)
        normalized = environment_name.lower().replace("_", "")

        if normalized == "meetingscheduling":
            return self.meeting_tools.get_tool_names()
        elif normalized == "personalassistant":
            return self.personal_assistant_tools.get_tool_names()
        elif normalized == "smartgrid":
            return self.smartgrid_tools.get_tool_names()
        else:
            return set()

    def get_blackboard_tool_names(self) -> Set[str]:
        """
        Get the set of tool names that this blackboard manager supports.

        Returns:
            Set of supported tool names
        """
        return {"get_blackboard_events", "post_message"}
         

    def get_tools_for_blackboard(self, phase: str) -> List[Dict[str, Any]]:
        """Get blackboard specific tools for the given phase. This is different from Environment tools."""
        # Define base tools available in all phases
        base_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_blackboard_events",
                    "description": "Get all events from a sepcific blackboard",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blackboard_id": {"type": "integer", "description": "ID of the blackboard you are getting information from"}
                        },
                        "required": ["blackboard_id"]
                    }
                }
            }
        ]

        # Add phase-specific tools
        if phase == "planning":
            planning_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "post_message",
                        "description": "Post a communication message to agents on the blackboard",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string", "description": "The message to communicate to other agents"},
                                "blackboard_id": {"type": "integer", "description": "ID of the blackboard you are posting to"}
                            },
                            "required": ["message"]
                        }
                    }
                },
            ]
            return base_tools + planning_tools

        return base_tools        
