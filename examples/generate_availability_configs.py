"""
Generate availability arrays for all parameter combinations.

This script generates availability arrays for AgentA and AgentB across all
combinations of the following parameters:
1. Negotiation Window: 1, 7, 14, 30 days
2. Time Slot: 15, 30, 60 minutes
3. Security parameter size: 128, 256 bits
4. Availability Density: 0%, 10%, 20%, 30%
5. Intersection Density: 0%, 5%, 10%

Total combinations: 4 × 3 × 2 × 4 × 3 = 288 configurations
"""

import json
import csv
import random
import itertools
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Configuration Parameters
# ============================================================================

NEGOTIATION_WINDOWS = [1, 7, 14, 30]  # days
TIME_SLOTS = [15, 30, 60]  # minutes (meeting interval duration)
SECURITY_PARAMS = [128, 256]  # bits
AVAILABILITY_DENSITIES = [0.0, 0.10, 0.20, 0.30]  # 0%, 10%, 20%, 30%
INTERSECTION_DENSITIES = [0.0, 0.05, 0.10]  # 0%, 5%, 10%

WORK_HOURS = 8  # Working hours per day (8:00-16:00 = 8 hours)
AGENTS = ["AgentA", "AgentB"]
SEED = 42  # Base seed for reproducibility


# ============================================================================
# Helper Functions
# ============================================================================

def generate_availability_arrays(
    window_days: int,
    time_slot_minutes: int,
    availability_density: float,
    intersection_density: float,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Generate availability arrays for all agents with guaranteed intersections.
    
    Logic:
    1. Array length = window_days × (WORK_HOURS × 60 / time_slot_minutes)
       Example: 1 day, 15 min → 1 × (8×60/15) = 32 slots
                7 days, 30 min → 7 × (8×60/30) = 112 slots
    2. Each array has availability_density fraction of 1s (rounded)
    3. Of these 1s, intersection_density fraction are COMMON in both agents (rounded)
    
    Args:
        window_days: Negotiation window in days
        time_slot_minutes: Meeting interval duration in minutes (15, 30, or 60)
        availability_density: Fraction of total slots that are available (0.0-1.0)
        intersection_density: Fraction of total slots that are common intersections (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping agent names to their availability arrays
    """
    rng = random.Random(seed)
    
    # Step 1: Calculate array dimensions
    # total_slots = window_days × (working_hours_per_day × 60 / time_slot_minutes)
    slots_per_day = (WORK_HOURS * 60) // time_slot_minutes
    total_slots = window_days * slots_per_day
    
    # Round fractional values to nearest integer
    num_available_per_agent = round(total_slots * availability_density)  # How many 1s each agent has
    num_intersections = round(total_slots * intersection_density)  # How many common 1s
    
    # Validate: intersection cannot exceed availability
    if num_intersections > num_available_per_agent:
        num_intersections = num_available_per_agent
    
    # Step 2: Select guaranteed intersection slots (common to both agents)
    if num_intersections > 0:
        intersection_slots = sorted(rng.sample(range(total_slots), num_intersections))
    else:
        intersection_slots = []
    
    availability = {}
    used_slots_by_agents = {}  # Track which slots each agent uses (excluding intersections)
    
    # Step 3: Generate availability for each agent
    for agent_idx, agent_name in enumerate(AGENTS):
        # Use agent-specific seed for variation
        agent_seed = seed + hash(agent_name) % 10000
        agent_rng = random.Random(agent_seed)
        
        # Start with all slots marked as BUSY (0)
        slots = [0] * total_slots
        
        # Set intersection slots to AVAILABLE (1) - these are COMMON
        for slot_idx in intersection_slots:
            slots[slot_idx] = 1
        
        # Calculate how many additional 1s this agent needs (beyond intersections)
        num_additional = num_available_per_agent - num_intersections
        
        if num_additional > 0:
            # Start with remaining slots (not in intersection)
            remaining_slots = [i for i in range(total_slots) if i not in intersection_slots]
            
            # CRITICAL: Exclude slots already used by previous agents to prevent overlap
            # Only intersection_slots should be common!
            if agent_idx > 0:
                for prev_agent_slots in used_slots_by_agents.values():
                    remaining_slots = [s for s in remaining_slots if s not in prev_agent_slots]
            
            # Ensure we don't try to sample more than available
            num_additional = min(num_additional, len(remaining_slots))
            
            if num_additional > 0:
                additional_slots = agent_rng.sample(remaining_slots, num_additional)
                for slot_idx in additional_slots:
                    slots[slot_idx] = 1
                
                # Track used slots (excluding intersections, since those are shared)
                used_slots_by_agents[agent_name] = additional_slots
        
        availability[agent_name] = slots
    
    return availability, intersection_slots


def generate_all_configurations() -> List[Dict]:
    """
    Generate all parameter combinations and their corresponding availability arrays.
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    config_id = 0
    
    # Generate all combinations
    for days, time_slot, security, avail_density, inter_density in itertools.product(
        NEGOTIATION_WINDOWS,
        TIME_SLOTS,
        SECURITY_PARAMS,
        AVAILABILITY_DENSITIES,
        INTERSECTION_DENSITIES
    ):
        config_id += 1
        
        # Total slots = window_days × (WORK_HOURS × 60 / time_slot_minutes)
        # Example: 1 day × (8×60/15) = 32 slots
        #          7 days × (8×60/30) = 112 slots
        slots_per_day = (WORK_HOURS * 60) // time_slot
        total_slots = days * slots_per_day
        
        # Generate availability arrays
        availability, intersection_slots = generate_availability_arrays(
            window_days=days,
            time_slot_minutes=time_slot,
            availability_density=avail_density,
            intersection_density=inter_density,
            seed=SEED + config_id  # Unique seed per configuration
        )
        
        # Calculate statistics
        agentA_available = sum(availability["AgentA"])
        agentB_available = sum(availability["AgentB"])
        actual_intersections = sum(
            1 for i in range(total_slots)
            if availability["AgentA"][i] == 1 and availability["AgentB"][i] == 1
        )
        
        config = {
            "config_id": config_id,
            "params": {
                "negotiation_window_days": days,
                "time_slot_minutes": time_slot,
                "security_bits": security,
                "availability_density": avail_density,
                "intersection_density": inter_density
            },
            "computed": {
                "slots_per_day": slots_per_day,
                "total_slots": total_slots,
                "guaranteed_intersections": len(intersection_slots),
                "guaranteed_intersection_slots": intersection_slots
            },
            "availability": availability,
            "statistics": {
                "agentA_available_count": agentA_available,
                "agentA_available_percentage": round(agentA_available / total_slots * 100, 2),
                "agentB_available_count": agentB_available,
                "agentB_available_percentage": round(agentB_available / total_slots * 100, 2),
                "actual_intersections": actual_intersections,
                "actual_intersection_percentage": round(actual_intersections / total_slots * 100, 2)
            }
        }
        
        configurations.append(config)
    
    return configurations


def save_configurations(configurations: List[Dict], output_dir: str = "outputs"):
    """
    Save configurations to JSON and CSV files.
    
    Args:
        configurations: List of configuration dictionaries
        output_dir: Output directory path
    """
    output_path = Path(output_dir) / "availability_configurations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # CSV FILE - Main output format
    # ========================================================================
    csv_file = output_path / "availability_configurations.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow([
            'config_id',
            'negotiation_window_days',
            'time_slot_minutes',
            'security_bits',
            'availability_density',
            'intersection_density',            'slots_per_day',            'total_slots',
            'slots_per_day',
            'agentA_available_count',
            'agentB_available_count',
            'actual_intersections',
            'AgentA',
            'AgentB'
        ])
        
        # Data rows
        for config in configurations:
            p = config["params"]
            c = config["computed"]
            s = config["statistics"]
            avail = config["availability"]
            
            # Convert availability arrays to string representation
            agentA_array = str(avail["AgentA"])
            agentB_array = str(avail["AgentB"])
            
            writer.writerow([
                config['config_id'],
                p['negotiation_window_days'],
                p['time_slot_minutes'],
                p['security_bits'],
                p['availability_density'],
                p['intersection_density'],
                c['slots_per_day'],
                c['total_slots'],
                c['guaranteed_intersections'],
                s['agentA_available_count'],
                s['agentB_available_count'],
                s['actual_intersections'],
                agentA_array,
                agentB_array
            ])
    
    print(f"✅ Saved CSV file with {len(configurations)} configurations to: {csv_file}")
    
    # ========================================================================
    # JSON FILE - Complete data with all metadata
    # ========================================================================
    all_configs_file = output_path / "all_configurations.json"
    with open(all_configs_file, 'w') as f:
        json.dump(configurations, f, indent=2)
    
    print(f"✅ Saved JSON file to: {all_configs_file}")
    
    # ========================================================================
    # SUMMARY FILE
    # ========================================================================
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AVAILABILITY CONFIGURATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Configurations: {len(configurations)}\n\n")
        
        f.write("Parameter Ranges:\n")
        f.write(f"  Negotiation Window: {NEGOTIATION_WINDOWS} days\n")
        f.write(f"  Time Slots: {TIME_SLOTS} minutes\n")
        f.write(f"  Security Parameters: {SECURITY_PARAMS} bits\n")
        f.write(f"  Availability Densities: {[f'{d*100:.0f}%' for d in AVAILABILITY_DENSITIES]}\n")
        f.write(f"  Intersection Densities: {[f'{d*100:.0f}%' for d in INTERSECTION_DENSITIES]}\n")
        f.write(f"\n")
        
        f.write("Sample Configurations:\n")
        f.write("-" * 80 + "\n")
        for config in configurations[:10]:  # First 10 configs
            p = config["params"]
            c = config["computed"]
            s = config["statistics"]
            f.write(f"\nConfig #{config['config_id']}: ")
            f.write(f"{p['negotiation_window_days']}d, ")
            f.write(f"{p['time_slot_minutes']}min, ")
            f.write(f"{p['security_bits']}bit, ")
            f.write(f"avail={p['availability_density']*100:.0f}%, ")
            f.write(f"inter={p['intersection_density']*100:.0f}%\n")
            f.write(f"  Total slots: {c['total_slots']}\n")
            f.write(f"  Guaranteed intersections: {c['guaranteed_intersections']}\n")
            f.write(f"  AgentA available: {s['agentA_available_count']}/{c['total_slots']} ({s['agentA_available_percentage']}%)\n")
            f.write(f"  AgentB available: {s['agentB_available_count']}/{c['total_slots']} ({s['agentB_available_percentage']}%)\n")
            f.write(f"  Actual intersections: {s['actual_intersections']} ({s['actual_intersection_percentage']}%)\n")
    
    print(f"✅ Saved summary to: {summary_file}")
    
    # ========================================================================
    # INDIVIDUAL JSON FILES (optional, for easier access)
    # ========================================================================
    individual_dir = output_path / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    for config in configurations:
        config_file = individual_dir / f"config_{config['config_id']:03d}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ Saved {len(configurations)} individual JSON files to: {individual_dir}")


def print_sample_configurations(configurations: List[Dict], num_samples: int = 5):
    """
    Print sample configurations to console.
    
    Args:
        configurations: List of configuration dictionaries
        num_samples: Number of samples to print
    """
    print("\n" + "=" * 80)
    print("SAMPLE CONFIGURATIONS")
    print("=" * 80 + "\n")
    
    for config in configurations[:num_samples]:
        p = config["params"]
        c = config["computed"]
        s = config["statistics"]
        avail = config["availability"]
        
        print(f"Config #{config['config_id']}:")
        print(f"  Parameters: {p['negotiation_window_days']}d, {p['time_slot_minutes']}min, "
              f"{p['security_bits']}bit, avail={p['availability_density']*100:.0f}%, "
              f"inter={p['intersection_density']*100:.0f}%")
        print(f"  Total Slots: {c['total_slots']} ({p['negotiation_window_days']}d × {c['slots_per_day']}slots/day)")
        print(f"    Formula: {p['negotiation_window_days']} × (8h×60min/{p['time_slot_minutes']}min) = {c['total_slots']}")
        print(f"  Guaranteed Intersections: {c['guaranteed_intersections']} slots at indices {c['guaranteed_intersection_slots'][:5]}...")
        print(f"  AgentA: {s['agentA_available_count']}/{c['total_slots']} available ({s['agentA_available_percentage']}%)")
        print(f"    Array (first 20): {avail['AgentA'][:20]}...")
        print(f"  AgentB: {s['agentB_available_count']}/{c['total_slots']} available ({s['agentB_available_percentage']}%)")
        print(f"    Array (first 20): {avail['AgentB'][:20]}...")
        print(f"  Actual Intersections Found: {s['actual_intersections']} ({s['actual_intersection_percentage']}%)")
        print()
        print()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING AVAILABILITY CONFIGURATIONS")
    print("=" * 80)
    print()
    
    print("Parameters:")
    print(f"  Negotiation Windows: {NEGOTIATION_WINDOWS} days")
    print(f"  Time Slots: {TIME_SLOTS} minutes")
    print(f"  Security Parameters: {SECURITY_PARAMS} bits")
    print(f"  Availability Densities: {[f'{d*100:.0f}%' for d in AVAILABILITY_DENSITIES]}")
    print(f"  Intersection Densities: {[f'{d*100:.0f}%' for d in INTERSECTION_DENSITIES]}")
    print()
    
    total_combinations = (
        len(NEGOTIATION_WINDOWS) *
        len(TIME_SLOTS) *
        len(SECURITY_PARAMS) *
        len(AVAILABILITY_DENSITIES) *
        len(INTERSECTION_DENSITIES)
    )
    print(f"Total Combinations: {total_combinations}")
    print()
    
    # Generate all configurations
    print("Generating configurations...")
    configurations = generate_all_configurations()
    print(f"✅ Generated {len(configurations)} configurations")
    print()
    
    # Print samples
    print_sample_configurations(configurations, num_samples=3)
    
    # Save to files
    print("Saving to files...")
    save_configurations(configurations)
    print()
    
    print("=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: outputs/availability_configurations/")
    print(f"📊 Total configurations: {len(configurations)}")
    print(f"📄 Main file: all_configurations.json")
    print(f"📝 Summary: summary.txt")
    print(f"📂 Individual files: individual/config_*.json")


if __name__ == "__main__":
    main()
