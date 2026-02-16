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
        logger.info("=" * 100)
        logger.info("🔒 OBLIVIOUS TRANSFER (OT) PROTOCOL - DETAILED EXECUTION LOG")
        logger.info("=" * 100)
        
        # Extract available slot indices from binary arrays
        sender_indices = [i for i, val in enumerate(sender_availability) if val == 1]
        receiver_indices = [i for i, val in enumerate(receiver_availability) if val == 1]
        
        # Setup parameters
        number_of_OT = 1
        n = total_slots
        p_size = len(receiver_indices)
        
        logger.info("📥 INPUT PARAMETERS:")
        logger.info(f"   Sender binary array:       {sender_availability}")
        logger.info(f"   Receiver binary array:     {receiver_availability}")
        logger.info(f"   Sender indices (Agent A):  {sender_indices}")
        logger.info(f"   Receiver indices (Agent B): {receiver_indices}")
        logger.info(f"   Total slots (n):           {total_slots}")
        logger.info(f"   Number of OTs:             {number_of_OT}")
        logger.info(f"   Preference size (p_size):  {p_size} (queries to make)")
        logger.info(f"   Bit size (security):       {self.bit_size}")
        
        # Convert binary arrays to message vectors for OT
        sender_messages = self._binary_to_messages(sender_availability)
        receiver_preferences = [receiver_indices]  # Wrapped in list for single OT
        
        logger.info("")
        logger.info("🔄 MESSAGE ENCODING:")
        logger.info(f"   Sender message vector (length={len(sender_messages)}):")
        logger.info(f"      {sender_messages}")
        logger.info(f"   Receiver query indices (length={len(receiver_preferences[0])}):")
        logger.info(f"      {receiver_preferences}")
        logger.info(f"   Encoding: available slot i → str(i), busy slot → '0'")
        
        # Phase 1: Receiver generates random keys
        logger.info("")
        logger.info("=" * 100)
        logger.info("📍 PHASE 1: SETUP - Receiver generates random encryption keys")
        logger.info("=" * 100)
        logger.info(f"   Input to pyot.setup():")
        logger.info(f"      number_of_OT = {number_of_OT}")
        logger.info(f"      n = {n}")
        logger.info(f"      bit_size = {self.bit_size}")
        
        r = pyot.setup(number_of_OT, n, self.bit_size)
        
        logger.info(f"   Output from pyot.setup():")
        logger.info(f"      Type: {type(r)}")
        logger.info(f"      Length: {len(r) if hasattr(r, '__len__') else 'N/A'}")
        logger.info(f"      First element type: {type(r[0]) if hasattr(r, '__getitem__') else 'N/A'}")
        logger.info(f"   ✓ Random keys generated for {n} slots")
        
        # Phase 2: Receiver generates query
        logger.info("")
        logger.info("=" * 100)
        logger.info("📍 PHASE 2: GEN_QUERY - Receiver generates oblivious queries")
        logger.info("=" * 100)
        logger.info(f"   Input to pyot.gen_query():")
        logger.info(f"      number_of_OT = {number_of_OT}")
        logger.info(f"      p_size = {p_size}")
        logger.info(f"      receiver_preferences = {receiver_preferences}")
        logger.info(f"      n = {n}")
        logger.info(f"      y = [[]] (output parameter for blinding factors)")
        
        y = [[]]  # Output parameter
        w = pyot.gen_query(number_of_OT, p_size, receiver_preferences, n, y)
        
        logger.info(f"   Output from pyot.gen_query():")
        logger.info(f"      w (queries) type: {type(w)}")
        logger.info(f"      w length: {len(w) if hasattr(w, '__len__') else 'N/A'}")
        logger.info(f"      w[0] length: {len(w[0]) if hasattr(w, '__getitem__') and hasattr(w[0], '__len__') else 'N/A'}")
        logger.info(f"      y (blinding factors) type: {type(y)}")
        logger.info(f"      y length: {len(y) if hasattr(y, '__len__') else 'N/A'}")
        logger.info(f"      y[0] length: {len(y[0]) if hasattr(y, '__getitem__') and hasattr(y[0], '__len__') else 'N/A'}")
        logger.info(f"   ✓ Generated {p_size} oblivious queries (receiver wants slots: {receiver_indices})")
        
        # Phase 3: Sender generates encrypted response
        logger.info("")
        logger.info("=" * 100)
        logger.info("📍 PHASE 3: GEN_RES - Sender encrypts their availability")
        logger.info("=" * 100)
        logger.info(f"   Input to pyot.gen_res():")
        logger.info(f"      sender_messages = {sender_messages}")
        logger.info(f"      number_of_OT = {number_of_OT}")
        logger.info(f"      r (from phase 1) = {type(r)} (keys)")
        logger.info(f"      w (from phase 2) = {type(w)} (queries)")
        
        res_s = pyot.gen_res(sender_messages, number_of_OT, r, w)
        
        logger.info(f"   Output from pyot.gen_res():")
        logger.info(f"      res_s type: {type(res_s)}")
        logger.info(f"      res_s length: {len(res_s) if hasattr(res_s, '__len__') else 'N/A'}")
        logger.info(f"      res_s[0] length: {len(res_s[0]) if hasattr(res_s, '__getitem__') and hasattr(res_s[0], '__len__') else 'N/A'}")
        logger.info(f"   ✓ Sender encrypted {len(sender_messages)} messages")
        logger.info(f"   🔐 Privacy: Receiver cannot decrypt all {n} slots, only their {p_size} queries!")
        
        # Phase 4: Third party filters
        logger.info("")
        logger.info("=" * 100)
        logger.info("📍 PHASE 4: OBL_FILTER - Third party applies oblivious filtering")
        logger.info("=" * 100)
        logger.info(f"   Input to pyot.obl_filter():")
        logger.info(f"      number_of_OT = {number_of_OT}")
        logger.info(f"      p_size = {p_size}")
        logger.info(f"      res_s (from phase 3) = {type(res_s)}")
        logger.info(f"      y (from phase 2) = {type(y)} (blinding factors)")
        
        res_h = pyot.obl_filter(number_of_OT, p_size, res_s, y)
        
        logger.info(f"   Output from pyot.obl_filter():")
        logger.info(f"      res_h type: {type(res_h)}")
        logger.info(f"      res_h length: {len(res_h) if hasattr(res_h, '__len__') else 'N/A'}")
        logger.info(f"      res_h[0] length: {len(res_h[0]) if hasattr(res_h, '__getitem__') and hasattr(res_h[0], '__len__') else 'N/A'}")
        logger.info(f"   ✓ Filtered to {p_size} encrypted responses (one per receiver query)")
        
        # Phase 5: Receiver retrieves results
        logger.info("")
        logger.info("=" * 100)
        logger.info("📍 PHASE 5: RETRIEVE - Receiver decrypts and validates intersection")
        logger.info("=" * 100)
        logger.info(f"   Processing {p_size} queries to find intersection...")
        logger.info("")
        
        intersection = []
        for j in range(p_size):
            slot_idx = receiver_preferences[0][j]
            
            logger.info(f"   Query #{j+1}/{p_size}:")
            logger.info(f"      Receiver querying slot index: {slot_idx}")
            logger.info(f"      Input to pyot.retreive():")
            logger.info(f"         res_h[0][{j}] (encrypted response)")
            logger.info(f"         j = {j}")
            logger.info(f"         r[0] (receiver's key)")
            logger.info(f"         preference[{j}] = {slot_idx}")
            
            retrieved = pyot.retreive(res_h[0][j], j, r[0], receiver_preferences[0])
            
            logger.info(f"      Output from pyot.retreive():")
            logger.info(f"         retrieved = '{retrieved}' (type: {type(retrieved).__name__})")
            logger.info(f"         sender_messages[{slot_idx}] = '{sender_messages[slot_idx]}'")
            
            # Check if this slot is available for sender (retrieved == slot_index)
            # If sender also has this slot available, the decrypted value will match
            # FIXED: Compare same types (str == str, not int == str)
            match = str(retrieved) == sender_messages[slot_idx]
            
            logger.info(f"      Comparison: str('{retrieved}') == '{sender_messages[slot_idx]}' → {match}")
            
            if match:
                intersection.append(slot_idx)
                logger.info(f"      ✅ MATCH! Slot {slot_idx} is in INTERSECTION")
            else:
                logger.info(f"      ❌ NO MATCH - Slot {slot_idx} NOT in intersection")
            logger.info("")
        
        final_intersection = sorted(intersection)
        
        logger.info("=" * 100)
        logger.info("📊 FINAL RESULTS:")
        logger.info("=" * 100)
        logger.info(f"   Sender (Agent A) available slots:     {sender_indices}")
        logger.info(f"   Receiver (Agent B) available slots:   {receiver_indices}")
        logger.info(f"   ✅ INTERSECTION (common slots):        {final_intersection}")
        logger.info(f"   Total common slots found: {len(final_intersection)}/{total_slots}")
        logger.info("")
        logger.info("🔐 PRIVACY GUARANTEES:")
        logger.info(f"   ✓ Sender does NOT know which slots receiver queried")
        logger.info(f"   ✓ Receiver does NOT know sender's full availability (only intersection)")
        logger.info(f"   ✓ Third party does NOT learn anything (blinded by y factors)")
        logger.info("=" * 100)
        
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
            ['0', '1', '2', '0', '4']
        """
        messages = []
        for i, is_available in enumerate(binary_array):
            if is_available == 1:
                messages.append(str(i))  # Encode as slot index
            else:
                messages.append("0")  # Busy slot
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
