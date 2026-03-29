"""
plain_intersection.py
=====================
Privacy-FREE baseline for meeting-slot intersection.

This module provides a straightforward (non-OT) bitwise-AND approach to
computing the common available slots between two agents.  It is intentionally
kept SELF-CONTAINED – it imports nothing from the main Terrarium codebase –
so that it can serve as an independent, reproducible baseline in a benchmarking
paper.

API is intentionally mirrored on OTManager.compute_intersection so the
benchmark harness can call both with identical arguments.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def plain_intersection(
    agent_a: List[int],
    agent_b: List[int],
) -> List[int]:
    """
    Compute the intersection of two binary availability arrays without any
    cryptographic protocol.

    Both arrays must have identical length.  An index is in the intersection
    iff both entries equal 1 (bitwise AND).

    Parameters
    ----------
    agent_a : list[int]
        Binary availability vector for Agent A (0 = busy, 1 = available).
    agent_b : list[int]
        Binary availability vector for Agent B (0 = busy, 1 = available).

    Returns
    -------
    list[int]
        Sorted list of slot indices where both agents are available.

    Raises
    ------
    ValueError
        If the two arrays differ in length or contain values other than 0/1.

    Examples
    --------
    >>> plain_intersection([0, 1, 1, 0, 1], [1, 1, 0, 0, 1])
    [1, 4]
    >>> plain_intersection([0, 0, 0], [0, 0, 0])
    []
    >>> plain_intersection([1, 1, 1], [1, 1, 1])
    [0, 1, 2]
    """
    if len(agent_a) != len(agent_b):
        raise ValueError(
            f"Arrays must have equal length: got {len(agent_a)} vs {len(agent_b)}"
        )

    result: List[int] = []
    for idx, (a, b) in enumerate(zip(agent_a, agent_b)):
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError(
                f"Arrays must be binary (0/1). "
                f"Got agent_a[{idx}]={a}, agent_b[{idx}]={b}."
            )
        if a == 1 and b == 1:
            result.append(idx)

    return result


# ---------------------------------------------------------------------------
# Thin wrapper class – mirrors OTManager interface for the benchmark harness
# ---------------------------------------------------------------------------

class PlainIntersectionManager:
    """
    Drop-in replacement for ``OTManager`` that uses plain bitwise intersection.

    This class exists purely so the benchmark harness can call either backend
    through a uniform interface::

        mgr = PlainIntersectionManager()
        common = mgr.compute_intersection(agent_a_array, agent_b_array)
    """

    # The real OTManager accepts bit_size; we accept it and silently ignore it
    # so the harness can instantiate both managers with identical code.
    def __init__(self, bit_size: int = 128) -> None:
        self.bit_size = bit_size  # stored for interface parity; unused here

    def compute_intersection(
        self,
        sender_availability: List[int],
        receiver_availability: List[int],
        total_slots: int | None = None,   # accepted for interface parity
    ) -> List[int]:
        """
        Compute intersection without any OT protocol.

        Parameters
        ----------
        sender_availability : list[int]
            Binary availability array for the sender (Agent A).
        receiver_availability : list[int]
            Binary availability array for the receiver (Agent B).
        total_slots : int, optional
            Accepted for API parity with OTManager; derived from array length
            when not provided.

        Returns
        -------
        list[int]
            Sorted list of mutually available slot indices.
        """
        return plain_intersection(sender_availability, receiver_availability)
