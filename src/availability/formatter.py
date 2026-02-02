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
    phase: str = AvailabilityConstants.PHASE_PLANNING,
    meeting_intersections: Dict[str, List[int]] = None,
    meeting_info: Dict[str, Dict[str, any]] = None
) -> str:
    """
    Format agent availability as a human-readable table for the blackboard.
    Creates a separate table for each agent in paper format (Day × Slot).
    Optionally includes meeting-specific intersection tables.
    
    Args:
        agent_slots: Dictionary mapping agent names to their binary availability lists
                    (1 = Available/Free [no meetings], 0 = Busy [has meetings])
        num_days: Number of days to display (default: 1)
        num_slots_per_day: Number of time slots per day (default: 12)
        phase: Current simulation phase (for header)
        meeting_intersections: Optional dict mapping meeting IDs to intersection availability
        meeting_info: Optional dict with meeting details (title, participants)
        
    Returns:
        Formatted string table with separate table per agent
        
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

    table_lines = [
        f"\n=== AVAILABILITY TABLE ({phase} Phase) ===",
        "Legend: 1 = Available/Free (No Meetings), 0 = Busy (Has Meetings)",
        ""
    ]
    
    # Create separate table for each agent
    for agent_idx, (agent, slots) in enumerate(agent_slots.items()):
        if agent_idx > 0:
            table_lines.append("")  # Blank line between agent tables
        
        table_lines.append(f"Agent: {agent}")
        
        # Create header row with slot numbers
        slot_headers = [f"Slot {i+1}" for i in range(num_slots_per_day)]
        header = "        | " + " | ".join(f"{s:^6}" for s in slot_headers) + " |"
        separator = "-" * len(header)
        
        table_lines.append(header)
        table_lines.append(separator)
        
        # Create row for each day
        for day in range(num_days):
            day_label = f"Day {day + 1}"
            row_values = []
            
            for slot in range(num_slots_per_day):
                t = day * num_slots_per_day + slot
                # Get availability (default to 1 [available] if index out of bounds)
                val = slots[t] if t < len(slots) else AvailabilityConstants.AVAILABLE
                row_values.append(f"{val:^6}")
            
            row = f"{day_label:^8}| " + " | ".join(row_values) + " |"
            table_lines.append(row)
    
    # Add meeting intersection tables if provided
    if meeting_intersections and meeting_info:
        table_lines.append("")
        table_lines.append("=" * 70)
        table_lines.append("MEETING-SPECIFIC COMMON AVAILABILITY")
        table_lines.append("(Slots where ALL participants are available)")
        table_lines.append("=" * 70)
        
        for meeting_id, intersection_slots in meeting_intersections.items():
            info = meeting_info.get(meeting_id, {})
            title = info.get('title', meeting_id)
            participants = info.get('participants', [])
            
            table_lines.append("")
            table_lines.append(f"Meeting: {title}")
            table_lines.append(f"Participants: {', '.join(participants)}")
            
            # Create header row
            slot_headers = [f"Slot {i+1}" for i in range(num_slots_per_day)]
            header = "        | " + " | ".join(f"{s:^6}" for s in slot_headers) + " |"
            separator = "-" * len(header)
            
            table_lines.append(header)
            table_lines.append(separator)
            
            # Create row for each day
            for day in range(num_days):
                day_label = f"Day {day + 1}"
                row_values = []
                
                for slot in range(num_slots_per_day):
                    t = day * num_slots_per_day + slot
                    val = intersection_slots[t] if t < len(intersection_slots) else AvailabilityConstants.AVAILABLE
                    row_values.append(f"{val:^6}")
                
                row = f"{day_label:^8}| " + " | ".join(row_values) + " |"
                table_lines.append(row)
            row = f"{day_label:^8}| " + " | ".join(row_values) + " |"
            table_lines.append(row)
        
    return "\n".join(table_lines)
