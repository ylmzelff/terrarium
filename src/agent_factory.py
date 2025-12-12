"""
Agent factory to build agents based on provider configuration. It simply 
Centralises agent construction for runner scripts.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent import Agent


def build_agents(
    agent_names: Sequence[str],
    *,
    provider: str,
    provider_label: str,
    llm_config: Dict[str, Any],
    model_name: str,
    max_conversation_steps: int,
    tool_logger: Any,
    trajectory_logger: Any,
    environment_name: str,
    generation_params: Dict[str, Any],
    vllm_runtime: Any = None,
    make_agent: Optional[Callable[[Any, str, str], Agent]] = None,
    shuffle: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Agent]:
    """
    Build a list of Agents (or Agent subclasses) given agent names and provider config.

    This keeps runner scripts thin while avoiding environment<->LLM coupling.
    """
    if log_fn is None:
        log_fn = logging.info

    if make_agent is None:
        make_agent = lambda client, name, agent_model_name: Agent(
            client,
            name,
            agent_model_name,
            max_conversation_steps,
            tool_logger,
            trajectory_logger,
            environment_name,
            generation_params=generation_params,
        )

    agents: List[Agent] = []
    for name in agent_names:
        if provider == "vllm":
            if not vllm_runtime:
                raise ValueError("vllm_runtime is required when provider == 'vllm'")
            client, agent_model_name = vllm_runtime.create_client(name)
        else:
            from src.utils import get_client_instance

            client = get_client_instance(llm_config)
            agent_model_name = model_name

        log_fn(f"Initializing Agent: {name} with {provider_label} - {agent_model_name}")
        agent = make_agent(client, name, agent_model_name)
        agents.append(agent)

    if shuffle:
        random.shuffle(agents)

    return agents
