"""
Availability table formatter and validator.

This module provides functions to validate and format agent availability data
into human-readable tables for blackboard visualization.
"""

from typing import Dict, List
import logging

from .constants import AvailabilityConstants

logger = logging.getLogger(__name__)

# bu fonksiyon veriyi tabloya yazmadan önce format kontorlü yapar. 
def validate_availability_data(
    agent_slots: Dict[str, List[int]], 
    num_days: int, 
    num_slots_per_day: int
) -> None:
    """
    Validate availability data before formatting.
    
    Args:
        agent_slots: Dictionary mapping agent names to their availability lists
        num_days: Number of days
        num_slots_per_day: Number of time slots per day
        
    Raises:
        ValueError: If data is invalid
        TypeError: If data types are incorrect
    """
    if not agent_slots:
        raise ValueError("Empty agent_slots dictionary")
    
    if num_days <= 0 or num_slots_per_day <= 0:
        raise ValueError(f"Invalid dimensions: {num_days} days x {num_slots_per_day} slots")
    
    total_slots = num_days * num_slots_per_day
    
    for agent, slots in agent_slots.items():
        if not isinstance(slots, list):
            raise TypeError(f"Agent '{agent}' slots must be a list, got {type(slots)}")
        if len(slots) != total_slots:
            raise ValueError(
                f"Agent '{agent}' has {len(slots)} slots, expected {total_slots}"
            )
        if not all(s in [AvailabilityConstants.AVAILABLE, AvailabilityConstants.BUSY] for s in slots):
            raise ValueError(
                f"Agent '{agent}' has invalid slot values "
                f"(must be {AvailabilityConstants.AVAILABLE} or {AvailabilityConstants.BUSY})"
            )

# doğrulanmış availablity dosyasını string bir tabloya yazar. 
def format_availability_table(
    agent_slots: Dict[str, List[int]],
    num_days: int = AvailabilityConstants.DEFAULT_NUM_DAYS,
    num_slots_per_day: int = AvailabilityConstants.DEFAULT_SLOTS_PER_DAY,
    phase: str = AvailabilityConstants.PHASE_PLANNING
) -> str:
    """
    Format agent availability as a human-readable table for the blackboard.
    
    Args:
        agent_slots: Dictionary mapping agent names to their binary availability lists
                    (1 = Available/Free [no meetings], 0 = Busy [has meetings])
        num_days: Number of days to display (default: 1)
        num_slots_per_day: Number of time slots per day (default: 12)
        phase: Current simulation phase (for header)
        
    Returns:
        Formatted string table
        
    Raises:
        ValueError: If validation fails
        TypeError: If data types are incorrect
    """
    if not agent_slots:
        return AvailabilityConstants.NO_DATA_MESSAGE
    
    # Validate input data
    try:
        validate_availability_data(agent_slots, num_days, num_slots_per_day)
    except (ValueError, TypeError) as e:
        logger.error("Availability data validation failed: %s", e)
        raise

    agents = list(agent_slots.keys())
    
    # Create header row with time slots
    header = "Time Slot | " + " | ".join(f"{agent:^10}" for agent in agents)
    separator = "-" * len(header)
    
    table_lines = [
        f"\n=== AVAILABILITY TABLE ({phase} Phase) ===",
        "Legend: 1 = Available/Free (No Meetings), 0 = Busy (Has Meetings)",
        "",
        header,
        separator
    ]
    
    # Iterate through all time slots
    total_slots = num_days * num_slots_per_day
    
    for t in range(total_slots):
        day = t // num_slots_per_day + 1
        slot = t % num_slots_per_day
        
        # Format row: Time | Agent1 | Agent2 ...
        row_prefix = f"Day {day} S{slot:<2} | "
        
        agent_values = []
        for agent in agents:
            # Get availability (default to 1 [available] if index out of bounds)
            slots = agent_slots.get(agent, [])
            val = slots[t] if t < len(slots) else AvailabilityConstants.AVAILABLE
            agent_values.append(f"{val:^10}")
            
        row = row_prefix + " | ".join(agent_values)
        table_lines.append(row)
        
    return "\n".join(table_lines)
