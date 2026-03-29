"""
benchmark_ot_vs_plain.py
========================
SOTA benchmark:  OT-based privacy-preserving intersection  vs.
                 plain (non-private) bitwise-AND intersection.

For every configuration row in the availability_configurations.csv dataset
the script:
  1. Parses AgentA and AgentB binary arrays.
  2. Runs the plain (non-OT) intersection and records wall-clock time.
  3. Runs the OT-based intersection (if pyot is available) and records wall-clock time.
  4. Verifies that both methods return the same intersection.
  5. Writes per-row results to  tests/results/benchmark_results.csv
  6. Prints a summary table to stdout.

Usage
-----
    cd c:/Users/lenovo/Terrarium
    python tests/benchmark_ot_vs_plain.py

Optional flags
--------------
    --csv PATH          Path to the configurations CSV  (default: auto-detect)
    --max-rows N        Only benchmark the first N rows (default: all)
    --no-ot             Skip OT benchmarks (plain only)
    --warmup N          Warm-up repetitions before timing (default: 0)
    --repeat N          Timing repetitions per configuration (default: 1)
    --out PATH          Output CSV path (default: tests/results/benchmark_results.csv)
    --log-level LEVEL   Logging level: DEBUG | INFO | WARNING (default: WARNING)
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import mean, stdev
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path surgery: make sure the repo root and tests/ are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent   # …/Terrarium
TESTS_DIR = Path(__file__).resolve().parent           # …/Terrarium/tests
CRYPTO_DIR = REPO_ROOT / "crypto"                     # …/Terrarium/crypto (contains pyot.pyd)

for p in (str(REPO_ROOT), str(TESTS_DIR), str(CRYPTO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Local imports  (no changes to main codebase)
# ---------------------------------------------------------------------------
from plain_intersection import PlainIntersectionManager   # tests/plain_intersection.py

# Optional OT import
OT_AVAILABLE = False
OTManager = None
try:
    from crypto.ot_manager import OTManager as _OTManager  # type: ignore
    OTManager = _OTManager
    OT_AVAILABLE = True
except Exception as exc:  # ImportError, OSError, etc.
    logging.getLogger(__name__).warning("OT module unavailable: %s", exc)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

DEFAULT_CSV = REPO_ROOT / "examples" / "outputs" / "availability_configurations" / "availability_configurations.csv"
DEFAULT_OUT = TESTS_DIR / "results" / "benchmark_results.csv"


@dataclass
class ConfigRow:
    """One row of the availability_configurations CSV."""
    config_id: int
    negotiation_window_days: int
    time_slot_minutes: int
    security_bits: int
    availability_density: float
    intersection_density: float
    total_slots: int
    actual_intersections: int
    agent_a: List[int]
    agent_b: List[int]


@dataclass
class BenchmarkResult:
    """Timing + correctness result for one configuration."""
    config_id: int
    total_slots: int
    security_bits: int
    availability_density: float
    intersection_density: float
    actual_intersections: int
    # Plain
    plain_mean_s: float = 0.0
    plain_stdev_s: float = 0.0
    plain_intersection: str = ""
    plain_intersection_size: int = 0
    # OT
    ot_mean_s: float = float("nan")
    ot_stdev_s: float = float("nan")
    ot_intersection: str = ""
    ot_intersection_size: int = 0
    # Verification
    results_match: str = "N/A"
    error: str = ""


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> List[int]:
    """
    Parse a string like '[0, 1, 0, 1]' into a Python list of ints.
    The CSV rows are wrapped in extra quotes so we strip them first.
    """
    raw = raw.strip().strip('"').strip("'")
    return list(ast.literal_eval(raw))


def load_configurations(csv_path: Path) -> List[ConfigRow]:
    """
    Load and parse the availability_configurations CSV.

    The CSV stores each data row as a single outer-quoted cell with this structure:
        config_id,days,slot_min,sec_bits,avail_dens,inter_dens,spd,total_slots,
        spd2,cntA,cntB,actual_intersections,"[a0, a1, ...]","[b0, b1, ...]"

    We use a regex to reliably extract the two bracket arrays and the preceding
    scalar fields regardless of minor quoting variations.
    """
    import re

    # Pattern: capture everything before the first "[", then the two arrays
    ARRAY_RE = re.compile(r'"?\[([^\]]+)\]"?')

    configs: List[ConfigRow] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header

        for raw_row in reader:
            # Reconstruct the full line content (handles 1-cell and multi-cell)
            full = ",".join(raw_row).strip()

            # Extract the two arrays
            arrays = ARRAY_RE.findall(full)
            if len(arrays) < 2:
                logger.debug("Row has <2 arrays, skipping: %s", full[:80])
                continue

            # The LAST two array matches are AgentA and AgentB
            try:
                agent_a = [int(x.strip()) for x in arrays[-2].split(",")]
                agent_b = [int(x.strip()) for x in arrays[-1].split(",")]
            except ValueError as exc:
                logger.debug("Array parse error: %s", exc)
                continue

            # Strip the two array sections to get the scalar prefix
            scalar_part = ARRAY_RE.sub("", full)          # remove arrays
            scalar_part = scalar_part.replace('"', "")    # remove stray quotes
            # Remove trailing commas that remain after array removal
            scalar_part = re.sub(r",+$", "", scalar_part.strip())
            meta = scalar_part.split(",")

            try:
                configs.append(ConfigRow(
                    config_id=int(meta[0]),
                    negotiation_window_days=int(meta[1]),
                    time_slot_minutes=int(meta[2]),
                    security_bits=int(meta[3]),
                    availability_density=float(meta[4]),
                    intersection_density=float(meta[5]),
                    total_slots=int(meta[7]),
                    actual_intersections=int(meta[11]),
                    agent_a=agent_a,
                    agent_b=agent_b,
                ))
            except (IndexError, ValueError) as exc:
                logger.debug("Scalar parse error: %s | row: %s", exc, meta)
                continue

    logger.info("Loaded %d configurations from %s", len(configs), csv_path)
    return configs


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_call(fn, *args, repeat: int = 1) -> Tuple[List[float], object]:
    """
    Call fn(*args) `repeat` times and return (list_of_durations_seconds, last_result).
    Uses perf_counter for sub-millisecond accuracy.
    """
    durations: List[float] = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return durations, result


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def benchmark_one(
    cfg: ConfigRow,
    plain_mgr: PlainIntersectionManager,
    ot_mgr,          # OTManager instance or None
    repeat: int = 1,
    warmup: int = 0,
) -> BenchmarkResult:
    """
    Benchmark both methods on a single configuration row.
    """
    n = len(cfg.agent_a)
    res = BenchmarkResult(
        config_id=cfg.config_id,
        total_slots=n,
        security_bits=cfg.security_bits,
        availability_density=cfg.availability_density,
        intersection_density=cfg.intersection_density,
        actual_intersections=cfg.actual_intersections,
    )

    # ---- Plain intersection ------------------------------------------------
    try:
        # warm-up
        for _ in range(warmup):
            plain_mgr.compute_intersection(cfg.agent_a, cfg.agent_b)
        # timed runs
        plain_times, plain_result = _time_call(
            plain_mgr.compute_intersection, cfg.agent_a, cfg.agent_b, repeat=repeat
        )
        res.plain_mean_s = mean(plain_times)
        res.plain_stdev_s = stdev(plain_times) if len(plain_times) > 1 else 0.0
        res.plain_intersection = str(plain_result)
        res.plain_intersection_size = len(plain_result)
    except Exception as exc:
        res.error += f"[PLAIN ERROR] {exc}  "
        logger.error("Config %d: plain intersection failed: %s", cfg.config_id, exc)

    # ---- OT intersection ---------------------------------------------------
    if ot_mgr is not None:
        try:
            # warm-up
            for _ in range(warmup):
                ot_mgr.compute_intersection(cfg.agent_a, cfg.agent_b, total_slots=n)
            # timed runs
            ot_times, ot_result = _time_call(
                ot_mgr.compute_intersection,
                cfg.agent_a, cfg.agent_b, n,
                repeat=repeat,
            )
            res.ot_mean_s = mean(ot_times)
            res.ot_stdev_s = stdev(ot_times) if len(ot_times) > 1 else 0.0
            res.ot_intersection = str(ot_result)
            res.ot_intersection_size = len(ot_result)
        except Exception as exc:
            res.error += f"[OT ERROR] {exc}  "
            logger.error("Config %d: OT intersection failed: %s", cfg.config_id, exc)

    # ---- Verification ------------------------------------------------------
    if res.plain_intersection and res.ot_intersection:
        try:
            plain_set = set(ast.literal_eval(res.plain_intersection))
            ot_set = set(ast.literal_eval(res.ot_intersection))
            res.results_match = "YES" if plain_set == ot_set else "NO"
        except Exception:
            res.results_match = "PARSE_ERROR"

    return res


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_RESULT_FIELDS = [f.name for f in fields(BenchmarkResult)]


def write_results_csv(results: List[BenchmarkResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            row = {f: getattr(r, f) for f in _RESULT_FIELDS}
            # Round floats for readability
            for key in ("plain_mean_s", "plain_stdev_s", "ot_mean_s", "ot_stdev_s"):
                v = row[key]
                try:
                    row[key] = f"{v:.9f}"
                except (TypeError, ValueError):
                    pass
            writer.writerow(row)
    print(f"\n✅  Results written to: {out_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print a rich summary table to stdout."""
    valid = [r for r in results if not r.error or r.plain_mean_s > 0]
    if not valid:
        print("No valid results to summarise.")
        return

    # Column widths
    W = dict(cfg=7, slots=7, sec=5, dens=7, plain_ms=12, ot_ms=12, speedup=10, match=7)

    def hdr(name: str, w: int) -> str:
        return name.center(w)

    sep = "-" * (sum(W.values()) + len(W) * 3)

    print()
    print("=" * len(sep))
    print("  BENCHMARK SUMMARY  —  OT vs Plain Intersection")
    print("=" * len(sep))
    hline = (
        f"  {'CfgID':>{W['cfg']}} | {'Slots':>{W['slots']}} | "
        f"{'Sec':>{W['sec']}} | {'Dens':>{W['dens']}} | "
        f"{'Plain (ms)':>{W['plain_ms']}} | {'OT (ms)':>{W['ot_ms']}} | "
        f"{'Speedup':>{W['speedup']}} | {'Match':>{W['match']}}"
    )
    print(hline)
    print("  " + sep)

    for r in valid:
        plain_ms = r.plain_mean_s * 1000
        ot_ms = r.ot_mean_s * 1000 if r.ot_mean_s == r.ot_mean_s else float("nan")  # nan check
        try:
            speedup = f"{ot_ms / plain_ms:.1f}×" if plain_ms > 0 and ot_ms == ot_ms else "N/A"
        except ZeroDivisionError:
            speedup = "N/A"

        ot_ms_str = f"{ot_ms:.3f}" if ot_ms == ot_ms else "N/A"
        print(
            f"  {r.config_id:>{W['cfg']}} | "
            f"{r.total_slots:>{W['slots']}} | "
            f"{r.security_bits:>{W['sec']}} | "
            f"{r.availability_density:>{W['dens']}.2f} | "
            f"{plain_ms:>{W['plain_ms']}.4f} | "
            f"{ot_ms_str:>{W['ot_ms']}} | "
            f"{speedup:>{W['speedup']}} | "
            f"{r.results_match:>{W['match']}}"
        )

    print("  " + sep)

    # Aggregate stats
    plain_vals = [r.plain_mean_s for r in valid if r.plain_mean_s > 0]
    ot_vals = [r.ot_mean_s for r in valid if r.ot_mean_s == r.ot_mean_s]

    if plain_vals:
        print(f"\n  Plain  — mean: {mean(plain_vals)*1000:.4f} ms  "
              f"| total: {sum(plain_vals)*1000:.2f} ms  "
              f"| n={len(plain_vals)}")
    if ot_vals:
        print(f"  OT     — mean: {mean(ot_vals)*1000:.4f} ms  "
              f"| total: {sum(ot_vals)*1000:.2f} ms  "
              f"| n={len(ot_vals)}")
        if plain_vals:
            overall_speedup = mean(ot_vals) / mean(plain_vals)
            print(f"  OT is {overall_speedup:.1f}× slower than plain on average.")

    mismatch = [r for r in valid if r.results_match == "NO"]
    if mismatch:
        print(f"\n  ⚠️  MISMATCH DETECTED in {len(mismatch)} configuration(s):")
        for r in mismatch:
            print(f"     config_id={r.config_id}  plain={r.plain_intersection}  ot={r.ot_intersection}")
    else:
        n_checked = len([r for r in valid if r.results_match in ("YES", "NO")])
        if n_checked:
            print(f"\n  ✅  All {n_checked} checked configs: plain == OT intersection.")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark OT-based vs. plain slot-intersection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Path to availability_configurations.csv")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Limit benchmark to first N rows")
    p.add_argument("--no-ot", action="store_true",
                   help="Skip OT benchmarks (plain only)")
    p.add_argument("--warmup", type=int, default=0,
                   help="Warm-up repetitions before timing")
    p.add_argument("--repeat", type=int, default=1,
                   help="Timing repetitions per configuration")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output CSV path")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ---- Load data ---------------------------------------------------------
    csv_path: Path = args.csv
    if not csv_path.exists():
        print(f"❌  CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    configs = load_configurations(csv_path)
    if not configs:
        print("❌  No configurations loaded. Check CSV format.", file=sys.stderr)
        sys.exit(1)

    if args.max_rows:
        configs = configs[: args.max_rows]

    print(f"\n{'='*60}")
    print(f"  Benchmark: OT vs Plain Intersection")
    print(f"  Configurations : {len(configs)}")
    print(f"  Repeat (timed) : {args.repeat}")
    print(f"  Warm-up runs   : {args.warmup}")
    print(f"  OT available   : {OT_AVAILABLE and not args.no_ot}")
    print(f"{'='*60}\n")

    # ---- Instantiate managers ----------------------------------------------
    plain_mgr = PlainIntersectionManager()

    ot_mgr = None
    if OT_AVAILABLE and not args.no_ot:
        try:
            ot_mgr = OTManager()  # uses default bit_size=128
            print("🔒 OT manager initialised (pyot available)\n")
        except Exception as exc:
            print(f"⚠️  Could not initialise OT manager: {exc}\n")
            ot_mgr = None
    elif args.no_ot:
        print("ℹ️  --no-ot flag set; skipping OT benchmarks.\n")
    else:
        print("ℹ️  pyot not available; running plain-only benchmark.\n")

    # ---- Run benchmarks ----------------------------------------------------
    results: List[BenchmarkResult] = []
    n_total = len(configs)
    bar_width = 40

    for i, cfg in enumerate(configs, 1):
        # Progress bar
        filled = int(bar_width * i / n_total)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {i}/{n_total}  cfg={cfg.config_id}  slots={len(cfg.agent_a)}", end="", flush=True)

        r = benchmark_one(
            cfg,
            plain_mgr=plain_mgr,
            ot_mgr=ot_mgr,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        results.append(r)

    print()  # newline after progress bar

    # ---- Output ------------------------------------------------------------
    print_summary(results)
    write_results_csv(results, args.out)


if __name__ == "__main__":
    main()
