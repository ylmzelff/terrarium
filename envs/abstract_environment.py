"""
Abstract base class for environments.

This module defines the interface that all environments must implement to work
with the CommunicationProtocol. It separates environment-specific logic from
the generic communication and phase management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from src.logger import BlackboardLogger, PromptLogger
from collections.abc import Mapping

import logging
logger = logging.getLogger(__name__)


class AbstractEnvironment(ABC):
    """
    Abstract base class for environments that can be used with CommunicationProtocol.

    This interface defines the minimal set of methods that any environment must
    implement to integrate with the communication protocol system. The protocol
    handles phases, iterations, blackboard management, and LLM interactions,
    while the environment handles domain-specific logic, actions, and context.

    Attributes:
        blackboard_logger: Optional logger for tracking blackboard state changes.
            Environments should initialize this in their initialize() method if they
            want blackboard logging. Set to None to disable logging.
        prompt_logger: Optional logger for tracking agent prompts (system and user).
            Environments should initialize this in their initialize() method if they
            want prompt logging. Set to None to disable logging.
    """

    # Standard attributes - environments should initialize these in initialize()
    blackboard_logger: Optional[BlackboardLogger] = None
    prompt_logger: Optional[PromptLogger] = None

    def get_agent_names(self) -> List[str]:
        """
        Get the list of all agent names in this environment.

        Returns:
            List of agent name strings

        This is used by the protocol to iterate through agents during phases.
        """
        agent_names = getattr(self, "agent_names", None)
        if agent_names is not None:
            return list(agent_names)

        message = f"{self.__class__.__name__} must set `self.agent_names` or override get_agent_names()"
        logger.error(message)
        raise NotImplementedError(message)

    def set_agent_clients(self, agents: List[Any]) -> None:
        """
        Set the instantiated agent clients for the environment.

        The protocol typically calls this after constructing the LLM-backed agents.
        Environments can override if they need additional bookkeeping.
        """
        self.agents = agents

    @abstractmethod
    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            **kwargs: Additional context (planning_round, etc.)

        Returns:
            Dictionary with agent context including:
            - Agent state (budget, inventory, utilities)
            - Environment state (store, available items)
            - Phase-specific information
            - Any other relevant context

        The protocol will pass this context to get_user_prompt along with
        blackboard contexts and available tools.
        """
        pass

    @abstractmethod
    def done(self, iteration: int) -> bool:
        """
        Check if the simulation is complete.

        Args:
            iteration: Current iteration number

        Returns:
            True if the simulation should stop, False to continue.
        """
        pass

    @abstractmethod
    def compute_max_joint_reward(self) -> float:
        """
        Compute the maximum joint reward achievable in this environment.

        Returns:
            Optimal joint reward as a float
        """
        pass

    @abstractmethod
    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """
        Compute a joint reward given agent actions.
        Can be a function of all agents' individual rewards using agent_reward().

        Args:
            actions: A mapping from agent names to their actions
        """
        pass

    @abstractmethod
    def agent_reward(self, agent_name: str, action: Any) -> float:
        """
        Compute the reward for a single agent given its action.
        Allows credit assignment of agent actions to joint reward.

        Args:
            agent_name: Name of the agent
            action: The action taken by the agent
        """
        pass

    def get_final_summary(self) -> Dict[str, Any]:
        """
        Get a final summary of the entire simulation.

        Returns:
            Dictionary with final simulation results

        Default implementation returns empty summary. Environments can override
        to provide meaningful final statistics and results.
        """
        return {}

    def generate_final_summary(self) -> Dict[str, Any]:
        """Log and return a final summary of the simulation."""
        env_logger = logging.getLogger(self.__class__.__module__)
        env_logger.info("%s", "=" * 60)
        env_logger.info("SIMULATION COMPLETE - FINAL SUMMARY")
        env_logger.info("%s", "=" * 60)

        final_summary = self.get_final_summary()

        current_iteration = getattr(self, "current_iteration", None)
        if current_iteration is not None:
            env_logger.info("Total iterations: %s", current_iteration)

        if final_summary:
            for key, value in final_summary.items():
                if isinstance(value, dict):
                    env_logger.info("%s:", key)
                    for sub_key, sub_value in value.items():
                        env_logger.info("  %s: %s", sub_key, sub_value)
                else:
                    env_logger.info("%s: %s", key, value)

        return final_summary

    def log_iteration_summary(self, iteration: int) -> None:
        """Log a summary header for an iteration then delegate to log_iteration()."""
        env_logger = logging.getLogger(self.__class__.__module__)
        env_logger.info("--- ITERATION %s SUMMARY ---", iteration)
        log_iteration = getattr(self, "log_iteration", None)
        if not callable(log_iteration):
            message = f"{self.__class__.__name__} must implement log_iteration() or override log_iteration_summary()"
            env_logger.error(message)
            raise NotImplementedError(message)
        log_iteration(iteration)
