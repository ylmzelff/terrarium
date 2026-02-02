# Availability Module - Technical Documentation

## 📋 Table of Contents

- [Overview](#overview)
- [What is Availability?](#what-is-availability)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Data Flow](#data-flow)
- [File Structure](#file-structure)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Integration Points](#integration-points)

---

## 🎯 Overview

The **Availability Module** is a core component of the Meeting Scheduling environment that generates, validates, and formats agent availability tables for multi-agent coordination. It provides agents with visibility into when other agents are busy or free, enabling more effective coordination during the planning phase.

**Key Features:**

- ✅ Binary availability representation (1=Free, 0=Busy)
- ✅ Multi-blackboard support
- ✅ Validation and error handling
- ✅ Human-readable table formatting
- ✅ Production-ready code quality

---

## 📊 What is Availability?

**Availability** represents each agent's schedule as a binary array:

- **1 = Available/Free** - Agent has NO meetings in this time slot
- **0 = Busy/In Meeting** - Agent has meetings in this time slot

### Example

Given these meetings:

```python
meetings = [
    Meeting(id="m001", start=1, end=3, participants=["Alice", "Bob"]),
    Meeting(id="m002", start=5, end=7, participants=["Alice", "Charlie"]),
    Meeting(id="m003", start=8, end=12, participants=["Alice", "Bob", "Charlie"])
]
```

**Timeline: 12 slots (0-11)**

```
Slot:      0  1  2  3  4  5  6  7  8  9 10 11
           ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
Alice:     ✓  ✗  ✗  ✓  ✓  ✗  ✗  ✓  ✗  ✗  ✗  ✗
           │  └──┘     │  └──┘  │  └─────────┘
           │  m001     │  m002  │    m003

Bob:       ✓  ✗  ✗  ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗
           │  └──┘           │  └─────────┘
           │  m001           │    m003

Charlie:   ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✓  ✗  ✗  ✗  ✗
                       │  └──┘  │  └─────────┘
                       │  m002  │    m003
```

**Binary Representation:**

```python
{
    "Alice":   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    "Bob":     [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "Charlie": [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
}
```

---

## 🏗️ Architecture

### Module Structure

```
src/availability/
├── __init__.py           # Public API exports
├── constants.py          # Configuration constants
├── formatter.py          # Table formatting and validation
├── test_availability.py  # Unit tests
└── README.md            # This file
```

### Design Principles

1. **Separation of Concerns**: Each file has a single responsibility
2. **Type Safety**: Full type hints for all functions
3. **Validation**: Input validation before processing
4. **Error Handling**: Proper logging and exception handling
5. **Testability**: Comprehensive unit test coverage

---

## ⚙️ How It Works

### Phase 1: Meeting Generation (Before Availability)

**When:** During `MeetingSchedulingEnvironment.__init__()`

```python
# 1. Read configuration
num_meetings = config["environment"]["num_meetings"]      # 3
timeline_length = config["environment"]["timeline_length"] # 12
num_agents = config["communication_network"]["num_agents"] # 2

# 2. Create CoLLAB config
collab_cfg = MeetingSchedulingConfig(
    num_agents=2,
    num_meetings=3,
    timeline_length=12,
    min_participants=2,
    max_participants=4,
    soft_meeting_ratio=0.6,
    rng_seed=42  # ← Deterministic randomness
)

# 3. Generate instance with meetings
instance = generate_instance(collab_cfg, instance_dir)

# Result: instance.meetings contains Meeting objects
```

**Meeting Generation Logic:**

```python
def _generate_meetings(config, agents, rng):
    meetings = []
    for idx in range(config.num_meetings):
        # Random meeting type
        is_soft = rng.random() < config.soft_meeting_ratio

        # Random time window
        if is_soft:
            start = rng.randrange(0, timeline_length - 3)
            duration = rng.randint(2, 4)
            end = start + duration
        else:
            start = rng.randrange(0, timeline_length - 1)
            end = start + 1

        # Random participants
        size = rng.randint(min_participants, max_participants)
        participants = rng.sample(agents, size)

        meetings.append(Meeting(id, type, title, start, end, participants))

    return meetings
```

**Important:** Meetings are **ALREADY** generated before availability module runs!

---

### Phase 2: Availability Generation

**When:** During `environment.async_init()` (before first planning round)

#### Step 1: Generate Availability Slots

```python
def _generate_availability_slots(self) -> Dict[str, List[int]]:
    """
    Convert meeting data to binary availability arrays.
    """
    total_slots = self.num_days * self.slots_per_day  # 1 * 12 = 12
    agent_slots = {}

    for agent_name in self.agent_names:
        # Start with all slots AVAILABLE (1)
        slots = [AvailabilityConstants.AVAILABLE] * total_slots

        # Mark BUSY (0) for each meeting the agent participates in
        for meeting in self.instance.meetings:
            if agent_name in meeting.participants:
                for t in range(meeting.start, meeting.end):
                    if 0 <= t < total_slots:
                        slots[t] = AvailabilityConstants.BUSY  # 0

        agent_slots[agent_name] = slots

    return agent_slots
```

**Algorithm:**

1. Initialize all slots to 1 (AVAILABLE)
2. For each meeting:
   - If agent participates → mark slots as 0 (BUSY)
3. Return binary arrays

---

#### Step 2: Multi-Blackboard Logging

```python
async def _async_log_availability_table(self):
    """
    Log availability to ALL relevant blackboards.
    """
    # Generate availability data
    agent_slots = self._generate_availability_slots()

    # Get all blackboards
    blackboards = self.communication_protocol.megaboard.blackboards

    # Log to each blackboard
    for blackboard in blackboards:
        # Filter agents for this blackboard
        relevant_agents = [a for a in agent_slots.keys()
                          if a in blackboard.agents]

        if not relevant_agents:
            continue

        filtered_slots = {agent: agent_slots[agent]
                         for agent in relevant_agents}

        # Log via MCP
        await client.call_tool("log_availability_table", {
            "blackboard_id": blackboard_id,
            "agent_slots": filtered_slots,
            "num_days": self.num_days,
            "num_slots_per_day": self.slots_per_day,
            "phase": AvailabilityConstants.PHASE_PLANNING
        })
```

**Features:**

- ✅ Logs to ALL blackboards (not just blackboard 0)
- ✅ Filters agents per blackboard
- ✅ Async MCP communication
- ✅ Error handling and logging

---

#### Step 3: Formatting and Validation

```python
def format_availability_table(agent_slots, num_days, num_slots_per_day, phase):
    """
    Create human-readable table from binary data.
    """
    # 1. Validate input
    validate_availability_data(agent_slots, num_days, num_slots_per_day)

    # 2. Create header
    header = "Time Slot | " + " | ".join(f"{agent:^10}" for agent in agents)

    # 3. Create rows
    for t in range(total_slots):
        day = t // num_slots_per_day + 1
        slot = t % num_slots_per_day
        row = f"Day {day} S{slot:<2} | " + values_for_all_agents

    # 4. Return formatted table
    return table_string
```

**Output Example:**

```
=== AVAILABILITY TABLE (planning Phase) ===
Legend: 1 = Available/Free (No Meetings), 0 = Busy (Has Meetings)

Time Slot |   Alice    |    Bob
-----------------------------------
Day 1 S0  |     1      |     1
Day 1 S1  |     0      |     0
Day 1 S2  |     0      |     0
Day 1 S3  |     1      |     1
...
```

---

## 🔄 Data Flow

### Complete Simulation Flow

```
1. base_main.py
   └─> await run_simulation(config)
       └─> environment = create_environment(...)

2. MeetingSchedulingEnvironment.__init__()
   ├─> Read config (num_meetings, timeline_length, etc.)
   ├─> collab_cfg = MeetingSchedulingConfig(...)
   ├─> instance = generate_instance(collab_cfg)
   │   └─> CoLLAB Problem Layer
   │       ├─> _generate_meetings() ← MEETINGS CREATED HERE
   │       ├─> _build_problem()
   │       └─> Save to outputs/collab_instances/
   └─> self.instance.meetings ← Meetings now available

3. base_main.py
   └─> await environment.async_init()

4. MeetingSchedulingEnvironment.async_init()
   └─> await self._async_log_availability_table()

5. _async_log_availability_table()
   ├─> agent_slots = self._generate_availability_slots()
   │   └─> Loop through self.instance.meetings
   │       └─> Mark busy slots for each agent
   │
   └─> For each blackboard:
       ├─> Filter agents
       └─> await client.call_tool("log_availability_table", ...)

6. MCP Server (src/server.py)
   └─> @mcp.tool() log_availability_table()
       └─> megaboard.log_availability_table(...)

7. Megaboard (src/blackboard.py)
   ├─> formatted_table = format_availability_table(...)
   │   └─> src/availability/formatter.py
   │       ├─> validate_availability_data()
   │       └─> Create table string
   │
   └─> self.post_system_message(kind="context", iteration=0)

8. Planning Phase Starts
   └─> Agents read blackboard context
       └─> AVAILABILITY TABLE VISIBLE ✅
```

### Timing

| Phase                      | Action                 | Availability Status  |
| -------------------------- | ---------------------- | -------------------- |
| Environment `__init__()`   | Meetings generated     | ❌ Not created yet   |
| Environment `async_init()` | Availability logged    | ✅ Created & logged  |
| Planning Round 1           | Agents read blackboard | ✅ Visible to agents |
| Execution Phase            | Agents make decisions  | ✅ Still available   |

**Key Point:** Availability is logged **ONCE** at `async_init()`, **BEFORE** any planning rounds start.

---

## 📁 File Structure

### Core Files

#### `constants.py`

```python
class AvailabilityConstants:
    # Event types
    EVENT_TYPE = "availability_table"
    EVENT_KIND = "context"

    # Slot values
    AVAILABLE = 1  # Free
    BUSY = 0       # In meeting

    # Defaults
    DEFAULT_NUM_DAYS = 1
    DEFAULT_SLOTS_PER_DAY = 12
```

#### `formatter.py`

- `validate_availability_data()` - Input validation
- `format_availability_table()` - Human-readable formatting

#### `__init__.py`

```python
from .constants import AvailabilityConstants
from .formatter import format_availability_table, validate_availability_data

__all__ = ["AvailabilityConstants", "format_availability_table", "validate_availability_data"]
```

---

## 📚 API Reference

### `validate_availability_data()`

```python
def validate_availability_data(
    agent_slots: Dict[str, List[int]],
    num_days: int,
    num_slots_per_day: int
) -> None:
```

**Purpose:** Validate availability data before formatting.

**Parameters:**

- `agent_slots`: Dictionary mapping agent names to availability lists
- `num_days`: Number of days
- `num_slots_per_day`: Number of time slots per day

**Raises:**

- `ValueError`: If data is invalid (empty, wrong size, invalid values)
- `TypeError`: If data types are incorrect

**Validation Checks:**

1. ✅ Non-empty agent_slots
2. ✅ Positive dimensions (num_days, num_slots_per_day)
3. ✅ Correct slot count (num_days × num_slots_per_day)
4. ✅ Valid slot values (only 0 or 1)
5. ✅ Correct data types (list of ints)

---

### `format_availability_table()`

```python
def format_availability_table(
    agent_slots: Dict[str, List[int]],
    num_days: int = AvailabilityConstants.DEFAULT_NUM_DAYS,
    num_slots_per_day: int = AvailabilityConstants.DEFAULT_SLOTS_PER_DAY,
    phase: str = AvailabilityConstants.PHASE_PLANNING
) -> str:
```

**Purpose:** Format agent availability as a human-readable table.

**Parameters:**

- `agent_slots`: Binary availability lists (1=Free, 0=Busy)
- `num_days`: Number of days to display (default: 1)
- `num_slots_per_day`: Number of slots per day (default: 12)
- `phase`: Current phase for header (default: "planning")

**Returns:** Formatted string table

**Example:**

```python
agent_slots = {
    "Alice": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    "Bob":   [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
}

table = format_availability_table(agent_slots, num_days=1, num_slots_per_day=12)
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m unittest src.availability.test_availability -v

# Expected output:
# test_constants_exist ... ok
# test_format_empty_slots ... ok
# test_format_invalid_data_raises ... ok
# test_validate_correct_data ... ok
# ... (14 tests total)
# OK
```

### Test Coverage

**TestAvailabilityValidation:**

- ✅ Empty agent slots
- ✅ Invalid dimensions
- ✅ Wrong slot length
- ✅ Invalid slot values
- ✅ Wrong data types
- ✅ Correct data passes

**TestAvailabilityFormatting:**

- ✅ Empty slots handling
- ✅ Single agent
- ✅ Multiple agents
- ✅ Multi-day formatting
- ✅ Legend inclusion
- ✅ Phase in header
- ✅ Invalid data rejection

**TestAvailabilityConstants:**

- ✅ All constants exist and have correct values

---

## 🔗 Integration Points

### Environment Integration

**File:** `envs/dcops/meeting_scheduling/meeting_scheduling_env.py`

```python
class MeetingSchedulingEnvironment(AbstractEnvironment):

    async def async_init(self):
        """Called after __init__, before planning starts."""
        await super().async_init()
        await self._async_log_availability_table()  # ← Integration point

    def _generate_availability_slots(self) -> Dict[str, List[int]]:
        """Generate slots from self.instance.meetings."""
        from src.availability import AvailabilityConstants
        # ... implementation

    async def _async_log_availability_table(self) -> None:
        """Log to all blackboards via MCP."""
        from src.availability import AvailabilityConstants
        # ... implementation
```

---

### Blackboard Integration

**File:** `src/blackboard.py`

```python
class Megaboard:

    def log_availability_table(self, blackboard_id, agent_slots,
                               num_days, num_slots_per_day, phase):
        """Format and log availability to blackboard."""
        from src.availability import format_availability_table, AvailabilityConstants

        formatted_table = format_availability_table(...)

        payload = {
            "message": formatted_table,
            "type": AvailabilityConstants.EVENT_TYPE,
            "data": agent_slots
        }

        self.post_system_message(
            blackboard_id,
            kind=AvailabilityConstants.EVENT_KIND,
            payload=payload
        )
```

---

### MCP Server Integration

**File:** `src/server.py`

```python
@mcp.tool()
def log_availability_table(blackboard_id: int, agent_slots: Dict[str, List[int]],
                          num_days: int = 1, num_slots_per_day: int = 12,
                          phase: str = "planning") -> Dict[str, str]:
    """MCP tool for logging availability tables."""
    try:
        megaboard.log_availability_table(blackboard_id, agent_slots,
                                        num_days, num_slots_per_day, phase)
        return {"status": "success", ...}
    except Exception as e:
        logger.error("Failed to log availability table: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}
```

---

## 🎯 Design Decisions

### Why Binary Representation?

**Alternatives Considered:**

1. ❌ String format ("busy"/"free") - More memory, harder to process
2. ❌ Meeting IDs per slot - Complex, hard to query
3. ✅ **Binary (0/1)** - Simple, efficient, easy to visualize

**Benefits:**

- Memory efficient
- Easy to aggregate (sum, count)
- Clear semantics
- Fast validation

---

### Why Multi-Blackboard Support?

**Problem:** Original implementation only logged to blackboard 0
**Solution:** Loop through all blackboards and filter agents

**Benefits:**

- Scales to complex network topologies
- Each blackboard sees relevant agents only
- Consistent with project architecture

---

### Why Validation?

**Problem:** Invalid data could cause formatting errors
**Solution:** Dedicated validation function with clear error messages

**Benefits:**

- Fail fast with clear errors
- Better debugging experience
- Type safety guarantees
- Production reliability

---

## 📝 Configuration

**File:** `examples/configs/meeting_scheduling.yaml`

```yaml
environment:
  num_meetings: 3 # Number of meetings to generate
  timeline_length: 12 # Total time slots
  min_participants: 2 # Min agents per meeting
  max_participants: 4 # Max agents per meeting
  soft_meeting_ratio: 0.6 # % of soft meetings

  # Availability table configuration
  num_days: 1 # Days to display in table
  slots_per_day: 12 # Slots per day (usually = timeline_length)
```

---

## 🚀 Future Enhancements

### Potential Improvements

1. **Meeting Info Addition**

   ```python
   # Add meeting details below availability table
   === MEETING SCHEDULE ===
   m001: Daily Standup [Slots 1-3] → Alice, Bob
   m002: Design Review [Slots 5-7] → Alice, Charlie
   ```

2. **Conflict Analysis**

   ```python
   === SLOT ANALYSIS ===
   • All agents free: [0, 3, 4, 7]
   • All agents busy: [1, 2, 8, 9, 10, 11]
   • Best coordination slots: [0, 3, 4]
   ```

3. **Visual Symbols**

   ```
   Day 1 S0  |     ✓      |     ✓
   Day 1 S1  |     ✗      |     ✗
   ```

4. **Time Labels**
   ```
   09:00-10:00 |     1      |     0
   10:00-11:00 |     0      |     1
   ```

---

## 📖 References

### Related Files

- `envs/dcops/meeting_scheduling/meeting_scheduling_env.py` - Environment integration
- `src/blackboard.py` - Blackboard logging
- `src/server.py` - MCP server tools
- `external/CoLLAB/problem_layer/meeting_scheduling/problem.py` - Meeting generation

### External Dependencies

- CoLLAB v2 problem layer
- FastMCP for tool communication
- Python 3.12+

---

## 👥 Contributing

When modifying the availability module:

1. ✅ Add type hints to all functions
2. ✅ Write docstrings (Google style)
3. ✅ Add unit tests for new features
4. ✅ Update this README
5. ✅ Follow project logging patterns
6. ✅ Use constants instead of magic strings

---

## 📄 License

This module is part of the Terrarium project. See main project LICENSE file.

---

## 🙏 Acknowledgments

Built on top of:

- CoLLAB v2 benchmark framework
- FastMCP for server communication
- Existing Terrarium infrastructure

---

**Last Updated:** February 2, 2026  
**Version:** 1.0.0  
**Author:** Terrarium Development Team
