from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import networkx as nx


@dataclass(frozen=True, slots=True)
class CommunicationNetwork:
    """A NetworkX-backed communication network.

    Convention used across Terrarium:
    - Nodes are agent names (strings).
    - Each edge corresponds to one blackboard with exactly the 2 endpoint agents.
    """
    
    # These are set in the constructor of subclasses
    graph: nx.Graph
    consolidate_channels_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.Graph):
            raise TypeError(f"graph must be a networkx.Graph, got: {type(self.graph)!r}")
        if not isinstance(self.consolidate_channels_enabled, bool):
            raise TypeError(
                "consolidate_channels_enabled must be a bool, "
                f"got: {type(self.consolidate_channels_enabled)!r}"
            )

        nodes = list(self.graph.nodes)
        if any(not isinstance(node, str) for node in nodes):
            raise TypeError("Network nodes must be agent name strings.")

        self_loops = [(u, v) for u, v in self.graph.edges if u == v]
        if self_loops:
            raise ValueError(f"Self-loop edges are not supported: {self_loops!r}")

    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    @property
    def agent_names(self) -> List[str]:
        return list(self.graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def validate_agents(self, agent_names: Sequence[str]) -> None:
        """Validate that the network nodes match the given agent names."""
        expected = set(agent_names)
        actual = set(self.graph.nodes)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "CommunicationNetwork nodes must match environment agent_names. "
                f"Missing={missing!r}, extra={extra!r}."
                f"This is likely an error in the environment logic."
            )

    def consolidate_channels(self) -> List[List[str]]:
        """Return a greedy vertex-disjoint clique cover as communication channels.
        Example: A complete graph with n nodes/agents --> one channel/blackboard with all n agents.

        Algorithm:
        - Repeatedly find a clique (size >= 3), create a multi-participant channel
          for it, remove those nodes, and repeat.
        - When no cliques (size >= 3) remain, create one channel per remaining edge.

        Notes:
        - This is an NP-hard problem, so finding an optimal solution is not feasible.
        - This algorithm may dominate computation time for large graphs.
        - Clique selection is greedy: pick the largest clique each iteration.
        - Channels are vertex-disjoint by construction (nodes are removed after selection).
        """
        working = self.graph.copy()
        channels: List[List[str]] = []

        while True:
            cliques = [sorted(c) for c in nx.find_cliques(working) if len(c) >= 3]
            if not cliques:
                break
            cliques.sort(key=lambda c: (-len(c), c))
            clique = cliques[0]
            channels.append(clique)
            working.remove_nodes_from(clique)

        edges: List[tuple[str, str]] = []
        for u, v in working.edges:
            a, b = sorted((u, v))
            edges.append((a, b))
        for a, b in sorted(edges):
            channels.append([a, b])

        return channels

    def channel_groups(self) -> List[List[str]]:
        """Return communication channel groups (blackboard participants) for this network."""
        if self.consolidate_channels_enabled:
            return self.consolidate_channels()

        edges: List[tuple[str, str]] = []
        for u, v in self.graph.edges:
            a, b = sorted((u, v))
            edges.append((a, b))
        return [[a, b] for a, b in sorted(edges)]
