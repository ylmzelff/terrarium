"""
benchmark_e2e.py
================
End-to-End (E2E) LLM Benchmark Runner.

This script runs the actual Terrarium LLM simulation (using examples/base_main.py)
but dynamically overrides the environment data to use the specific Agent A & Agent B
availability arrays from your CSV file.

Crucially, it does this using "monkey patching" — replacing python functions
in memory at runtime — so that NOT A SINGLE LINE of the main codebase is modified.

It will measure the total real-world time (including LLM generation, tool calls, etc.)
for two modes:
    1. WITHOUT OT (uses plain bitwise intersection)
    2. WITH OT (uses the cryptographic pyot implementation)

Usage
-----
    cd c:/Users/lenovo/Terrarium
    python tests/benchmark_e2e.py --max-rows 5
"""

import sys
import time
import argparse
import asyncio
import logging
from pathlib import Path

# Add project root to sys.path so we can import Terrarium modules seamlessly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the existing CSV parser from our fast benchmark script
from tests.benchmark_ot_vs_plain import load_configurations, DEFAULT_CSV
from tests.plain_intersection import plain_intersection

# Import the main simulation runner and config loader
from examples.base_main import run_simulation
from src.utils import load_config
from envs.dcops.meeting_scheduling.meeting_scheduling_env import MeetingSchedulingEnvironment
import crypto

logger = logging.getLogger(__name__)

# =======================================================================
# MONKEY PATCHING (Overriding main code dynamically without modifying files)
# =======================================================================

# 1. Save original functions so we can restore them later
ORIGINAL_GENERATE_AVAILABILITY = MeetingSchedulingEnvironment._generate_simulated_availability
ORIGINAL_COMPUTE_INTERSECTION = crypto.compute_private_intersection


def patch_environment_data(agent_a_array: list, agent_b_array: list):
    """
    Forces the LLM environment to use our CSV arrays when the LLM calls 'fetch_my_calendar'.
    """
    def mocked_generate(self, participants: list):
        # Depending on how the agents are named in the config. By default AgentA and AgentB.
        availability = {}
        for p in participants:
            if "A" in p:
                availability[p] = agent_a_array
            elif "B" in p:
                availability[p] = agent_b_array
            else:
                availability[p] = agent_a_array # Fallback
        
        logger.info(f"💉 E2E Benchmark: Injected CSV arrays into environment for {participants}")
        return availability

    MeetingSchedulingEnvironment._generate_simulated_availability = mocked_generate


def patch_crypto_module(use_ot: bool):
    """
    If use_ot=False, forces the entire simulation to use plain_intersection 
    instead of the heavy cryptographic OT.
    """
    if use_ot:
        # Restore the original OT implementation
        crypto.compute_private_intersection = ORIGINAL_COMPUTE_INTERSECTION
        logger.info("🔒 E2E Benchmark: Using original Crypto OT implementation")
    else:
        # Replace the OT implementation with our plain high-speed baseline
        def mocked_intersection(sender_availability, receiver_availability, total_slots=12):
            return plain_intersection(sender_availability, receiver_availability)
        
        crypto.compute_private_intersection = mocked_intersection
        logger.info("🔓 E2E Benchmark: MOCKED Crypto OT -> Using Plain Intersection Baseline")


# =======================================================================
# EXECUTION ENGINE
# =======================================================================

async def run_e2e_benchmark(csv_path: Path, config_path: Path, max_rows: int = None):
    # Load the LLM Simulation config
    base_config = load_config(str(config_path))
    
    # Load test scenarios
    configs = load_configurations(csv_path)
    if max_rows:
        configs = configs[:max_rows]

    print(f"\n============================================================")
    print(f"🚀 STARTING E2E LLM BENCHMARK")
    print(f"   Scenarios to run: {len(configs)}")
    print(f"   LLM Model: {base_config['llm']['model']}")
    print(f"============================================================\n")

    results = []

    for i, row in enumerate(configs, 1):
        print(f"\n" + "#"*70)
        print(f"📝 RUNNING SCENARIO {i}/{len(configs)} (Config ID: {row.config_id})")
        print(f"#"*70)

        # Inject this row's data into the simulation environment globally
        patch_environment_data(row.agent_a, row.agent_b)

        for use_ot in [False, True]:
            mode_name = "WITH OT" if use_ot else "WITHOUT OT (PLAIN)"
            print(f"\n---> 🧪 Executing Mode: {mode_name}")
            
            # Inject the appropriate algorithm (OT or Plain)
            patch_crypto_module(use_ot=use_ot)

            # Assign a unique run timestamp so blackboard logs don't overwrite
            base_config["simulation"]["run_timestamp"] = f"benchmark_cfg{row.config_id}_{'ot' if use_ot else 'plain'}"
            base_config["simulation"]["note"] = f"E2E Benchmark - Config {row.config_id} - {mode_name}"

            # Start the timer!
            start_time = time.perf_counter()
            
            try:
                # Actually run the LLMs
                success = await run_simulation(base_config)
            except Exception as e:
                print(f"❌ Simulation crashed: {e}")
                success = False

            # Stop the timer!
            end_time = time.perf_counter()
            duration_seconds = end_time - start_time

            print(f"\n⏱️  {mode_name} completed in: {duration_seconds:.2f} seconds")
            
            # Save metrics
            results.append({
                "config_id": row.config_id,
                "mode": mode_name,
                "duration_seconds": duration_seconds,
                "success": success
            })

    # Print final E2E Report
    print(f"\n============================================================")
    print(f"📊 END-TO-END BENCHMARK RESULTS SUMMARY")
    print(f"============================================================")
    for res in results:
        status = "✅" if res["success"] else "❌"
        print(f"Config {res['config_id']:>3} | {res['mode']:<20} | {res['duration_seconds']:>8.2f} sec | {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "examples/configs/meeting_scheduling.yaml")
    parser.add_argument("--max-rows", type=int, default=1, help="Number of scenarios to run (LLMs are slow, keep this small for testing)")
    args = parser.parse_args()

    # Create tests/results dir if it doesn't exist
    (REPO_ROOT / "tests/results").mkdir(exist_ok=True, parents=True)

    asyncio.run(run_e2e_benchmark(args.csv, args.config, args.max_rows))
