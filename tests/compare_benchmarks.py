"""
Compare OT vs Plain Benchmark Results
======================================
Loads timing results from separate OT and Plain benchmarks and generates
comparison analysis and plots.

Usage:
    python tests/compare_benchmarks.py
    python tests/compare_benchmarks.py --ot-file ot_timing_results.json --plain-file plain_timing_results.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def load_results(filepath: Path) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(ot_results: Dict, plain_results: Dict):
    """Compare and print OT vs Plain results."""
    print("=" * 80)
    print("BENCHMARK COMPARISON: OT Protocol vs Plain Intersection")
    print("=" * 80)
    print()
    
    # Create result lookup by size
    ot_by_size = {r['size']: r for r in ot_results['results']}
    plain_by_size = {r['size']: r for r in plain_results['results']}
    
    # Get common sizes
    common_sizes = sorted(set(ot_by_size.keys()) & set(plain_by_size.keys()))
    
    if not common_sizes:
        print("❌ No common array sizes found between OT and Plain results")
        return None
    
    print(f"Comparing {len(common_sizes)} array sizes: {common_sizes}")
    print()
    print("-" * 80)
    print(f"{'Size':>6} | {'Plain (ms)':>12} | {'OT (ms)':>12} | {'Overhead':>10} | {'Speedup':>10}")
    print("-" * 80)
    
    comparison_data = []
    
    for size in common_sizes:
        ot = ot_by_size[size]
        plain = plain_by_size[size]
        
        plain_avg_ms = plain['avg_time'] * 1000
        ot_avg_ms = ot['avg_time'] * 1000
        overhead = ot_avg_ms / plain_avg_ms if plain_avg_ms > 0 else float('inf')
        speedup = f"{overhead:.1f}x" if overhead != float('inf') else "∞"
        
        print(f"{size:6d} | {plain_avg_ms:12.6f} | {ot_avg_ms:12.6f} | {overhead:10.2f} | {speedup:>10}")
        
        comparison_data.append({
            "size": size,
            "plain_avg_ms": plain_avg_ms,
            "ot_avg_ms": ot_avg_ms,
            "overhead": overhead,
            "plain_std_ms": plain['std_dev'] * 1000,
            "ot_std_ms": ot['std_dev'] * 1000,
        })
    
    print("-" * 80)
    print()
    
    # Summary statistics
    overheads = [d['overhead'] for d in comparison_data if d['overhead'] != float('inf')]
    if overheads:
        avg_overhead = sum(overheads) / len(overheads)
        min_overhead = min(overheads)
        max_overhead = max(overheads)
        
        print("📊 Summary Statistics:")
        print(f"   Average OT overhead: {avg_overhead:.1f}x slower than plain")
        print(f"   Min overhead:        {min_overhead:.1f}x (size {comparison_data[overheads.index(min_overhead)]['size']})")
        print(f"   Max overhead:        {max_overhead:.1f}x (size {comparison_data[overheads.index(max_overhead)]['size']})")
    
    print()
    print("=" * 80)
    
    return comparison_data


def export_comparison(comparison_data: List[Dict], output_file: str = "comparison_results.csv"):
    """Export comparison to CSV."""
    output_path = project_root / "tests" / "results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("size,plain_avg_ms,ot_avg_ms,overhead,plain_std_ms,ot_std_ms\n")
        for data in comparison_data:
            f.write(f"{data['size']},{data['plain_avg_ms']:.6f},{data['ot_avg_ms']:.6f},"
                   f"{data['overhead']:.4f},{data['plain_std_ms']:.6f},{data['ot_std_ms']:.6f}\n")
    
    print(f"📁 Comparison exported to: {output_path}")
    return output_path


def plot_comparison(comparison_data: List[Dict]):
    """Plot comparison results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return
    
    sizes = [d['size'] for d in comparison_data]
    plain_times = [d['plain_avg_ms'] for d in comparison_data]
    ot_times = [d['ot_avg_ms'] for d in comparison_data]
    overheads = [d['overhead'] for d in comparison_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute times
    ax1.plot(sizes, plain_times, 'o-', label='Plain AND', linewidth=2, markersize=8, color='blue')
    ax1.plot(sizes, ot_times, 's-', label='OT Protocol', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Array Size (slots)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('OT vs Plain Intersection - Absolute Times', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Overhead
    ax2.plot(sizes, overheads, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Array Size (slots)', fontsize=12)
    ax2.set_ylabel('Overhead (OT / Plain)', fontsize=12)
    ax2.set_title('OT Protocol Overhead', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1, label='No overhead')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    output_path = project_root / "tests" / "results" / "comparison_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {output_path}")
    
    # Try to show (works in Jupyter/Colab)
    try:
        plt.show()
    except:
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare OT and Plain benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--ot-file",
        type=str,
        default="ot_timing_results.json",
        help="OT benchmark results file (default: ot_timing_results.json)"
    )
    parser.add_argument(
        "--plain-file",
        type=str,
        default="plain_timing_results.json",
        help="Plain benchmark results file (default: plain_timing_results.json)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (only show comparison table)"
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    results_dir = project_root / "tests" / "results"
    ot_path = results_dir / args.ot_file
    plain_path = results_dir / args.plain_file
    
    # Check if files exist
    if not ot_path.exists():
        print(f"❌ OT results file not found: {ot_path}")
        print(f"   Run: python tests/benchmark_ot_timing.py")
        sys.exit(1)
    
    if not plain_path.exists():
        print(f"❌ Plain results file not found: {plain_path}")
        print(f"   Run: python tests/benchmark_plain_timing.py")
        sys.exit(1)
    
    # Load results
    print(f"📂 Loading OT results from: {ot_path.name}")
    ot_results = load_results(ot_path)
    
    print(f"📂 Loading Plain results from: {plain_path.name}")
    plain_results = load_results(plain_path)
    print()
    
    # Compare
    comparison_data = compare_results(ot_results, plain_results)
    
    if comparison_data:
        # Export CSV
        export_comparison(comparison_data)
        
        # Plot (unless disabled)
        if not args.no_plot:
            plot_comparison(comparison_data)
        
        print()
        print("✅ Comparison complete!")


if __name__ == "__main__":
    main()
