"""
Constants for availability table functionality.
"""


class AvailabilityConstants:
    """Constants used in availability table generation and formatting."""
    
    # Event types
    EVENT_TYPE = "availability_table"
    EVENT_KIND = "context"
    
    # Phase names
    PHASE_PLANNING = "planning"
    PHASE_EXECUTION = "execution"
    
    # Defaults
    DEFAULT_NUM_DAYS = 1
    DEFAULT_SLOTS_PER_DAY = 12
    
    # Slot values
    AVAILABLE = 1  # Agent is free (no meetings)
    BUSY = 0       # Agent has meetings
    
    # Messages
    NO_DATA_MESSAGE = "No availability data."
    ERROR_MESSAGE_TEMPLATE = "Error logging availability table to blackboard {blackboard_id}: {error}"
