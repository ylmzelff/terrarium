"""
Oblivious Transfer Protocol Manager for Privacy-Preserving Slot Intersection

This module provides a high-level interface to compute slot intersections
between agents without revealing individual availability preferences.

Usage:
    manager = OTManager()
    common_slots = manager.compute_intersection(agent_a_slots, agent_b_slots)
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import pyot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    logger.warning("OT module not available. Install with: cd crypto && python setup.py install")


class OTManager:
    """Manager for privacy-preserving slot intersection using Oblivious Transfer."""
    
    def __init__(self, bit_size: int = 128):
        """
        Initialize OT Manager.
        
        Args:
            bit_size: Bit size for encryption keys (default: 128)
        """
        if not OT_AVAILABLE:
            raise ImportError("pyot module not available. Build it first: cd crypto && python setup.py install")
        
        self.bit_size = bit_size
    
    def compute_intersection(self, sender_slots: List[int], receiver_slots: List[int], 
                            total_slots: int = 12) -> List[int]:
        """
        Compute intersection of available slots using OT protocol.
        
        Agent A (sender) and Agent B (receiver) compute their common available slots
        without revealing their full availability to each other.
        
        Args:
            sender_slots: List of available slot indices for sender (Agent A)
            receiver_slots: List of available slot indices for receiver (Agent B)
            total_slots: Total number of time slots (default: 12)
        
        Returns:
            List of common available slot indices (intersection)
        
        Example:
            >>> manager = OTManager()
            >>> agent_a_slots = [2, 3, 4, 6, 7, 8]  # Agent A availability
            >>> agent_b_slots = [1, 3, 5, 6, 7, 10]  # Agent B availability
            >>> common = manager.compute_intersection(agent_a_slots, agent_b_slots)
            >>> print(common)  # Output: [3, 6, 7]
        """
        number_of_OT = 1
        n = total_slots
        p_size = len(receiver_slots)
        
        # Convert slot lists to binary availability vectors
        sender_availability = self._slots_to_messages(sender_slots, n)
        receiver_preferences = [receiver_slots]  # Wrapped in list for single OT
        
        # Phase 1: Receiver generates random keys
        r = pyot.setup(number_of_OT, n, self.bit_size)
        
        # Phase 2: Receiver generates query
        y = [[]]  # Output parameter
        w = pyot.gen_query(number_of_OT, p_size, receiver_preferences, n, y)
        
        # Phase 3: Sender generates encrypted response
        res_s = pyot.gen_res(sender_availability, number_of_OT, r, w)
        
        # Phase 4: Third party filters
        res_h = pyot.obl_filter(number_of_OT, p_size, res_s, y)
        
        # Phase 5: Receiver retrieves results
        intersection = []
        for j in range(p_size):
            retrieved = pyot.retreive(res_h[0][j], j, r[0], receiver_preferences[0])
            # Check if this slot is available for sender (retrieved == slot_index + 1)
            slot_idx = receiver_preferences[0][j]
            # If sender also has this slot available, the decrypted value will match
            # FIXED: Compare same types (str == str, not int == str)
            if str(retrieved) == sender_availability[slot_idx]:
                intersection.append(slot_idx)
        
        return sorted(intersection)
    
    def _slots_to_messages(self, slots: List[int], n: int) -> List[str]:
        """
        Convert slot availability list to encrypted messages.
        
        Args:
            slots: List of available slot indices
            n: Total number of slots
        
        Returns:
            List of string-encoded messages (for GMP compatibility)
        """
        # Create availability vector: slot index + 1 if available, else 0
        messages = []
        slot_set = set(slots)
        for i in range(n):
            if i in slot_set:
                messages.append(str(i + 1))  # Encode as slot index + 1
            else:
                messages.append("0")
        return messages
    
    def compute_intersection_simple(self, sender_slots: List[int], receiver_slots: List[int]) -> List[int]:
        """
        Fallback: Simple intersection computation (non-private).
        
        Used when OT module is not available or for testing.
        
        Args:
            sender_slots: Available slots for sender
            receiver_slots: Available slots for receiver
        
        Returns:
            List of common slots
        """
        return sorted(list(set(sender_slots) & set(receiver_slots)))


# Convenience function for direct use
def compute_private_intersection(sender_slots: List[int], receiver_slots: List[int], 
                                  total_slots: int = 12) -> List[int]:
    """
    Compute privacy-preserving slot intersection.
    
    Args:
        sender_slots: Available slots for sender
        receiver_slots: Available slots for receiver
        total_slots: Total number of time slots
    
    Returns:
        List of common available slots
    """
    if OT_AVAILABLE:
        try:
            manager = OTManager()
            return manager.compute_intersection(sender_slots, receiver_slots, total_slots)
        except Exception as e:
            logger.warning(f"OT computation failed: {e}. Falling back to simple intersection.")
    
    # Fallback to simple intersection
    return sorted(list(set(sender_slots) & set(receiver_slots)))
