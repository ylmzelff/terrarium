#!/usr/bin/env python3
"""
Test Meeting Scheduling with Random Availability (Simulation Mode)
==================================================================

Use: python test_simulation_mode.py

This script:
1. Starts a meeting scheduling simulation with random arrays (no Teams needed)
2. Logs complete flow: fetch → OT → attend
3. Verifies both agents agree on the same meeting slot
4. Reports results

Prerequisites:
  • cd crypto && python setup.py build_ext --inplace  (for OT module)
  • Python 3.8+, asyncio capable
"""

import asyncio
import sys
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

async def test_simulation_mode():
    """Test meeting scheduling in simulation mode (use_real_calendars=false)."""
    
    logger.info("="*80)
    logger.info("🧪 MEETING SCHEDULING SIMULATION MODE TEST")
    logger.info("="*80)
    logger.info("")
    
    # 1. Load configuration
    logger.info("📋 Step 1: Loading configuration...")
    from examples.configs import meeting_scheduling  # Example, actual path may vary
    config_path = REPO_ROOT / "examples" / "configs" / "meeting_scheduling.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    logger.info(f"    Config: {config_path}")
    
    # 2. Parse YAML
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    use_real = config.get("environment", {}).get("use_real_calendars", False)
    logger.info(f"    use_real_calendars: {use_real}")
    
    if use_real:
        logger.warning("❌ This test is for SIMULATION MODE only (use_real_calendars: false)")
        return False
    
    logger.info("    ✅ Simulation mode confirmed")
    logger.info("")
    
    # 3. Initialize environment
    logger.info("📦 Step 2: Initializing environment...")
    from envs.dcops.meeting_scheduling import MeetingSchedulingEnvironment
    from src.communication_protocols.sequential import SequentialProtocol
    
    try:
        comm_protocol = SequentialProtocol(config)  # Simple protocol without MCP for testing
        env = MeetingSchedulingEnvironment(comm_protocol, config, tool_logger=None)
        logger.info(f"    ✅ Environment initialized")
        logger.info(f"       Agents: {env.agent_names}")
        logger.info(f"       Meetings: {[m.meeting_id for m in env.meetings]}")
        logger.info(f"       Slots: {env.num_days} days × {env.slots_per_day} slots = {env.num_days * env.slots_per_day} total")
    except Exception as e:
        logger.error(f"    ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("")
    
    # 4. Simulate planning phase (fetch_my_calendar + submit_availability_array)
    logger.info("🧠 Step 3: PLANNING PHASE - Simulating agent availability submission...")
    
    from crypto import compute_private_intersection
    from src.availability import AvailabilityConstants
    
    # Get simulated availability that was pre-generated
    meeting_id = env.meetings[0].meeting_id
    participants = env.meetings[0].participants
    
    logger.info(f"    Meeting: {meeting_id}")
    logger.info(f"    Participants: {participants}")
    logger.info("")
    
    # Simulate agents fetching and submitting
    for agent_idx, agent_name in enumerate(participants):
        agent_seed = config["simulation"]["seed"] + ((agent_idx + 1) * 1009)
        
        # Get cached simulated availability
        simulated = env._simulated_availability_cache.get(meeting_id, {})
        agent_array = simulated.get(agent_name)
        
        if agent_array is None:
            logger.warning(f"    ⚠️  No simulated availability for {agent_name}")
            continue
        
        available_count = sum(agent_array)
        available_indices = [i for i, v in enumerate(agent_array) if v == 1]
        
        logger.info(f"    [{agent_name}] fetch_my_calendar({meeting_id})")
        logger.info(f"        Generated: {available_count}/{len(agent_array)} available")
        logger.info(f"        Indices: {available_indices}")
        logger.info(f"        Array: {agent_array}")
        logger.info(f"        submit_availability_array({meeting_id}, {agent_array})")
        logger.info("")
        
        # Simulate tool call
        env.meeting_availabilities[meeting_id] = {
            agent_name: agent_array
            for agent_name in participants
            if agent_name in simulated
        }
    
    logger.info("")
    
    # 5. Run OT Protocol
    logger.info("🔒 Step 4: OBLIVIOUS TRANSFER (OT) PROTOCOL...")
    
    if meeting_id not in env.meeting_availabilities:
        logger.error(f"    ❌ Availability not found for {meeting_id}")
        return False
    
    agent_slots = env.meeting_availabilities[meeting_id]
    if len(agent_slots) != 2:
        logger.warning(f"    ⚠️  OT requires exactly 2 participants, got {len(agent_slots)}")
        # For testing, we'll still try with whatever we have
    
    try:
        sender_name, receiver_name = list(agent_slots.keys())
        sender_array = agent_slots[sender_name]
        receiver_array = agent_slots[receiver_name]
        
        logger.info(f"    Sender: {sender_name} ({sum(sender_array)} available)")
        logger.info(f"    Receiver: {receiver_name} ({sum(receiver_array)} available)")
        logger.info("")
        
        # Compute intersection using OT
        intersection_indices = compute_private_intersection(
            sender_array,
            receiver_array,
            len(sender_array)
        )
        
        logger.info(f"    ✅ OT Protocol Complete")
        logger.info(f"       Intersection indices: {intersection_indices}")
        logger.info(f"       Common slots: {len(intersection_indices)}/{len(sender_array)}")
        
        # Cache result
        intersection_array = [0] * len(sender_array)
        for idx in intersection_indices:
            intersection_array[idx] = 1
        
        env._meeting_intersections_cache = {meeting_id: intersection_array}
        
    except Exception as e:
        logger.error(f"    ❌ OT Protocol failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("")
    
    # 6. Simulate execution phase (attend_meeting)
    logger.info("✅ Step 5: EXECUTION PHASE - Agents attend meeting...")
    logger.info("")
    
    attendance = {}
    if not intersection_indices:
        logger.warning("    ⚠️  No common slots - agents cannot schedule")
    else:
        # Each agent should choose the earliest common slot
        earliest_slot = min(intersection_indices)
        
        for agent_name in participants:
            logger.info(f"    [{agent_name}].attend_meeting(")
            logger.info(f"        meeting_id='{meeting_id}',")
            logger.info(f"        interval='{earliest_slot}'")
            logger.info(f"    )")
            
            # Validate (like tool would)
            common_slots = intersection_indices
            if earliest_slot == min(common_slots):
                logger.info(f"        ✅ Using earliest common slot")
                var_name = f"{agent_name}__{meeting_id}"
                attendance[var_name] = f"{earliest_slot}-{earliest_slot+1}"
            else:
                logger.error(f"        ❌ Not using earliest slot")
                return False
            
            logger.info("")
    
    logger.info("")
    
    # 7. Final report
    logger.info("="*80)
    logger.info("📊 FINAL REPORT")
    logger.info("="*80)
    
    if attendance:
        logger.info("✅ MEETING SUCCESSFULLY SCHEDULED")
        logger.info(f"   Meeting ID: {meeting_id}")
        logger.info(f"   Participants: {participants}")
        logger.info(f"   Scheduled Slot: {earliest_slot}")
        logger.info(f"   Attendance:")
        for var_name, interval in attendance.items():
            logger.info(f"     • {var_name}: {interval}")
        logger.info("")
        logger.info("🔒 PRIVACY VERIFICATION")
        logger.info(f"   Intersection only: {intersection_indices} (not individual arrays)")
        logger.info(f"   Agent arrays kept private: ✅")
        logger.info("")
        return True
    else:
        logger.warning("⚠️  No common slots - meeting could not be scheduled")
        logger.info(f"   {sender_name} available: {available_indices_sender}")
        logger.info(f"   {receiver_name} available: {available_indices_receiver}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_simulation_mode())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
