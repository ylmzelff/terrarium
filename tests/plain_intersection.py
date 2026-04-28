"""
plain_intersection.py
=====================
Privacy-FREE baseline for meeting-slot intersection.
Uses Python's array module (not list) for results.
"""
from __future__ import annotations
import array
from typing import List


def plain_intersection(
    agent_a: List[int],
    agent_b: List[int],
) -> array.array:
    """
    Bitwise AND ile iki binary availability array'in kesişim indekslerini döner.
    Sonuç: array.array (type code 'i')
    """
    if len(agent_a) != len(agent_b):
        raise ValueError(
            f"Array uzunluklari esit olmali: {len(agent_a)} != {len(agent_b)}"
        )
    result = array.array('i')
    for idx, (a, b) in enumerate(zip(agent_a, agent_b)):
        if a == 1 and b == 1:
            result.append(idx)
    return result


class PlainIntersectionManager:
    """OTManager ile ayni arayuz — benchmark harmless icin."""
    def __init__(self, bit_size: int = 128) -> None:
        self.bit_size = bit_size  # interface parity; kullanilmaz

    def compute_intersection(self, sender, receiver, total_slots=None) -> array.array:
        return plain_intersection(sender, receiver)
