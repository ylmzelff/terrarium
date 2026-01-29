from typing import Dict, List, Optional

def format_availability_table(
    agent_slots: Dict[str, List[int]],
    num_days: int = 1,
    num_slots_per_day: int = 12,
    phase: str = "Planning"
) -> str:
    """
    Format agent availability as a human-readable table for the blackboard.
    
    Args:
        agent_slots: Dictionary measuring agent names to their binary availability lists (1=Available, 0=Busy)
        num_days: Number of days to display (default: 1)
        num_slots_per_day: Number of time slots per day (default: 12)
        phase: Current simulation phase (for header)
        
    Returns:
        Formatted string table
    """
    if not agent_slots:
        return "No availability data."

    agents = list(agent_slots.keys())
    # Create header row with time slots
    header = "Time Slot | " + " | ".join(f"{agent:^10}" for agent in agents)
    separator = "-" * len(header)
    
    table_lines = [
        f"\n=== AVAILABILITY TABLE ({phase} Phase) ===",
        "Legend: 1 = Available (Has Meetings), 0 = Busy/No Meeting",
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
            # Get availability (default to 0 if index out of bounds)
            slots = agent_slots.get(agent, [])
            val = slots[t] if t < len(slots) else 0
            agent_values.append(f"{val:^10}")
            
        row = row_prefix + " | ".join(agent_values)
        table_lines.append(row)
        
    return "\n".join(table_lines)
