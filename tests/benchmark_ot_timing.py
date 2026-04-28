"""
OT Protocol Benchmark - Timing Test
====================================
Measures execution time of privacy-preserving Oblivious Transfer protocol
for different array sizes.

Usage:
    python tests/benchmark_ot_timing.py
    python tests/benchmark_ot_timing.py --sizes 8 16 32 --runs 50
    
Google Colab:
    !cd crypto && python setup.py install
    !python tests/benchmark_ot_timing.py
"""

import time
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import OT protocol
try:
    from crypto.ot_manager import OTManager
    OT_AVAILABLE = True
except ImportError as e:
    print(f"❌ ERROR: OT module not available: {e}")
    print("   Install with: cd crypto && python setup.py install")
    OT_AVAILABLE = False
    sys.exit(1)


class OTBenchmark:
    """Benchmark OT protocol intersection timing."""
    
    def __init__(self, array_sizes: List[int] = None, bit_size: int = 128):
        """
        Initialize OT benchmark.
        
        Args:
            array_sizes: List of array sizes to test
            bit_size: OT encryption bit size (128 or 256)
        """
        self.array_sizes = array_sizes or [8, 16, 32, 56, 112, 224, 240, 448, 480, 960]
        self.bit_size = bit_size
        self.results = []
        self.ot_manager = OTManager(bit_size=bit_size)
        
    def generate_arrays(self, size: int, pattern: str = "zeros") -> Tuple[List[int], List[int]]:
        """
        Generate test arrays.
        
        Args:
            size: Array size
            pattern: "zeros", "ones", "alternating", "random"
            
        Returns:
            Tuple of (array_a, array_b)
        """
        if pattern == "zeros":
            return [0] * size, [0] * size
        elif pattern == "ones":
            return [1] * size, [1] * size
        elif pattern == "alternating":
            return [1, 0] * (size // 2), [0, 1] * (size // 2)
        elif pattern == "random":
            import random
            return [random.randint(0, 1) for _ in range(size)], \
                   [random.randint(0, 1) for _ in range(size)]
        else:
            return [0] * size, [0] * size
    
    def benchmark_single(self, array_a: List[int], array_b: List[int], runs: int = 100) -> Dict:
        """
        Benchmark OT protocol for given arrays.
        
        Args:
            array_a: First array (sender)
            array_b: Second array (receiver)
            runs: Number of runs for averaging
            
        Returns:
            Dict with timing statistics
        """
        times = []
        size = len(array_a)
        
        print(f"   Running {runs} iterations...", end=" ", flush=True)
        
        for i in range(runs):
            start = time.perf_counter()
            result = self.ot_manager.compute_intersection(array_a, array_b, size)
            end = time.perf_counter()
            times.append(end - start)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)
        
        print("✓")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate standard deviation
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        
        return {
            "size": size,
            "runs": runs,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "times": times,
        }
    
    def run_benchmark(self, custom_arrays: Dict[int, Tuple[List[int], List[int]]] = None,
                     runs: int = 100, pattern: str = "zeros"):
        """
        Run full benchmark suite.
        
        Args:
            custom_arrays: Dict mapping size -> (array_a, array_b). If None, uses generated patterns.
            runs: Number of runs per test
            pattern: Pattern for generated arrays ("zeros", "ones", "alternating", "random")
        """
        print("=" * 80)
        print("OT PROTOCOL BENCHMARK")
        print("=" * 80)
        print(f"Bit size:     {self.bit_size}")
        print(f"Array sizes:  {self.array_sizes}")
        print(f"Runs per size: {runs}")
        if custom_arrays:
            print(f"Input:        Custom arrays")
        else:
            print(f"Input:        Generated ({pattern})")
        print("=" * 80)
        print()
        
        for size in self.array_sizes:
            print(f"📊 Testing size: {size}")
            
            # Get arrays
            if custom_arrays and size in custom_arrays:
                array_a, array_b = custom_arrays[size]
                print(f"   Using custom input arrays")
            else:
                array_a, array_b = self.generate_arrays(size, pattern)
                print(f"   Using {pattern} pattern")
            
            # Validate arrays
            if len(array_a) != size or len(array_b) != size:
                print(f"   ⚠️  Warning: Array size mismatch (expected {size}, got {len(array_a)}, {len(array_b)})")
                continue
            
            # Run benchmark
            result = self.benchmark_single(array_a, array_b, runs)
            self.results.append(result)
            
            # Print results
            print(f"   ⏱️  Avg:  {result['avg_time']*1000:.3f} ms")
            print(f"   📈 Min:  {result['min_time']*1000:.3f} ms")
            print(f"   📈 Max:  {result['max_time']*1000:.3f} ms")
            print(f"   📊 StdDev: {result['std_dev']*1000:.3f} ms")
            print()
        
        print("=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
    
    def export_results(self, filename: str = "ot_timing_results.json"):
        """
        Export results to JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = project_root / "tests" / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare export data
        export_data = {
            "bit_size": self.bit_size,
            "array_sizes": self.array_sizes,
            "results": []
        }
        
        for result in self.results:
            # Export without full times array (too large)
            export_result = {
                "size": result["size"],
                "runs": result["runs"],
                "avg_time": result["avg_time"],
                "min_time": result["min_time"],
                "max_time": result["max_time"],
                "std_dev": result["std_dev"],
            }
            export_data["results"].append(export_result)
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Results exported to: {output_path}")
        return output_path
    
    def export_csv(self, filename: str = "ot_timing_results.csv"):
        """Export results to CSV file."""
        output_path = project_root / "tests" / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("size,runs,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms\n")
            for result in self.results:
                f.write(f"{result['size']},{result['runs']},"
                       f"{result['avg_time']*1000:.6f},"
                       f"{result['min_time']*1000:.6f},"
                       f"{result['max_time']*1000:.6f},"
                       f"{result['std_dev']*1000:.6f}\n")
        
        print(f"📁 CSV exported to: {output_path}")
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark OT protocol timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/benchmark_ot_timing.py
  python tests/benchmark_ot_timing.py --sizes 8 16 32 --runs 50
  python tests/benchmark_ot_timing.py --pattern random --bit-size 256
        """
    )
    
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 56, 112, 224, 240, 448, 480, 960],
        help="Array sizes to test"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of runs per size (default: 100)"
    )
    parser.add_argument(
        "--pattern",
        choices=["zeros", "ones", "alternating", "random"],
        default="zeros",
        help="Array generation pattern (default: zeros)"
    )
    parser.add_argument(
        "--bit-size",
        type=int,
        choices=[128, 256],
        default=128,
        help="OT encryption bit size (default: 128)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ot_timing_results",
        help="Output filename prefix (default: ot_timing_results)"
    )
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = OTBenchmark(array_sizes=args.sizes, bit_size=args.bit_size)
    benchmark.run_benchmark(runs=args.runs, pattern=args.pattern)
    
    # Export results
    benchmark.export_results(f"{args.output}.json")
    benchmark.export_csv(f"{args.output}.csv")
    
    print()
    print("✅ OT Protocol benchmark complete!")


if __name__ == "__main__":
    main()
