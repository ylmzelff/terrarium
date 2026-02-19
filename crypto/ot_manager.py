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
    
    def compute_intersection(self, sender_availability: List[int], receiver_availability: List[int], 
                            total_slots: int = 12) -> List[int]:
        """
        Compute intersection of available slots using OT protocol.
        
        Agent A (sender) and Agent B (receiver) compute their common available slots
        without revealing their full availability to each other.
        
        Args:
            sender_availability: Binary array for sender (0=busy, 1=available)
            receiver_availability: Binary array for receiver (0=busy, 1=available)
            total_slots: Total number of time slots (default: 12)
        
        Returns:
            List of common available slot indices (intersection)
        
        Example:
            >>> manager = OTManager()
            >>> agent_a = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]  # Agent A availability
            >>> agent_b = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]  # Agent B availability
            >>> common = manager.compute_intersection(agent_a, agent_b)
            >>> print(common)  # Output: [3, 6, 7]
        """
        logger.info("=" * 80)
        logger.info("🔒 OT PROTOCOL EXECUTION (Compact Log)")
        logger.info("=" * 80)
        
        # Extract available slot indices from binary arrays
        sender_indices = [i for i, val in enumerate(sender_availability) if val == 1]
        receiver_indices = [i for i, val in enumerate(receiver_availability) if val == 1]
        
        # Setup parameters
        number_of_OT = 1
        n = total_slots
        p_size = len(receiver_indices)
        
        logger.info(f"📥 INPUT: Sender={sender_indices}, Receiver={receiver_indices}, n={total_slots}")
        
        # Convert binary arrays to message vectors for OT
        sender_messages = self._binary_to_messages(sender_availability)
        receiver_preferences = [receiver_indices]  # Wrapped in list for single OT
        
        # Phase 1: Setup
        r = pyot.setup(number_of_OT, n, self.bit_size)
        logger.info(f"✓ Phase 1: Setup ({n} keys generated)")
        
        # Phase 2: Gen Query
        w, y = pyot.gen_query(number_of_OT, p_size, receiver_preferences, n)
        logger.info(f"✓ Phase 2: GenQuery ({p_size} oblivious queries)")
        
        # Phase 3: Gen Response
        res_s = pyot.gen_res(sender_messages, number_of_OT, r, w)
        logger.info(f"✓ Phase 3: GenRes ({n} encrypted messages)")
        
        # Phase 4: Oblivious Filter
        res_h = pyot.obl_filter(number_of_OT, p_size, res_s, y)
        logger.info(f"✓ Phase 4: oblFilter ({p_size} filtered responses)")
        
        # Phase 5: Retrieve
        
        logger.info(f"✓ Phase 5: Retrieve (processing {p_size} queries)")
        
        intersection = []
        for j in range(p_size):
            slot_idx = receiver_preferences[0][j]
            retrieved = pyot.retreive(res_h[0][j], j, r[0], receiver_preferences[0])
            
            # Check if slot is in common (compare decrypted value with sender's message)
            match = str(retrieved) == sender_messages[slot_idx]
            
            if match:
                intersection.append(slot_idx)
        
        final_intersection = sorted(intersection)
        
        logger.info("=" * 80)
        logger.info(f"📊 RESULT: Intersection = {final_intersection} ({len(final_intersection)}/{total_slots} slots)")
        logger.info(f"🔐 Privacy: Both parties only know intersection, not each other's full availability")
        logger.info("=" * 80)
        
        return final_intersection
    
    def _binary_to_messages(self, binary_array: List[int]) -> List[str]:
        """
        Convert binary availability array (0/1) to encrypted messages for OT.
        
        Args:
            binary_array: Binary list where 1 = available, 0 = busy
        
        Returns:
            List of string-encoded messages (for GMP compatibility)
            
        Example:
            >>> _binary_to_messages([0, 1, 1, 0, 1])
            ['-1', '1', '2', '-1', '4']  # -1 = busy, index = available
        """
        messages = []
        for i, is_available in enumerate(binary_array):
            if is_available == 1:
                messages.append(str(i))  # Encode as slot index
            else:
                messages.append("-1")  # Busy slot (unique marker to avoid collision with index 0)
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
def compute_private_intersection(sender_availability: List[int], receiver_availability: List[int], 
                                  total_slots: int = 12) -> List[int]:
    """
    Compute privacy-preserving slot intersection using OT protocol.
    
    Args:
        sender_availability: Binary array of sender's availability (0=busy, 1=available)
        receiver_availability: Binary array of receiver's availability (0=busy, 1=available)
        total_slots: Total number of time slots
    
    Returns:
        List of common available slot indices
        
    Raises:
        ImportError: If OT module is not available
        Exception: If OT protocol fails
    """
    if not OT_AVAILABLE:
        raise ImportError(
            "OT module not available. Install with: python crypto/setup.py build_ext --inplace"
        )
    
    manager = OTManager()
    return manager.compute_intersection(sender_availability, receiver_availability, total_slots)
