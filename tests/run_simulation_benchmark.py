"""
run_simulation_benchmark.py
===========================
OT vs Plain AND — E2E Simulation Benchmark (zero arrays)

Her array size için:
  - [0]*N arrayler kullanılır (varyans sıfır, deterministik)
  - Bir OT simülasyonu, bir Plain simülasyonu çalıştırılır
  - LLM her ikisinde de aynı 4 tool call yapar: submit x2, attend x2
  - crypto_time_s (sadece OT/plain çağrısı) ve e2e süresi ayrı ayrı kaydedilir

Çıktı: tests/results/simulation_benchmark_TIMESTAMP.csv

Kullanım:
    python tests/run_simulation_benchmark.py
    python tests/run_simulation_benchmark.py --sizes 8 32 112 --runs 3
    python tests/run_simulation_benchmark.py --no-ot   # sadece plain
"""

from __future__ import annotations

import asyncio
import copy
import csv
import gc
import logging
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_logging, load_config
from dotenv import load_dotenv

SIZES    = [960, 480, 448, 240, 224, 112, 56, 32, 16, 8]
BASE_CFG = str(PROJECT_ROOT / "examples" / "configs" / "meeting_scheduling.yaml")

KNOWN_MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(size: int, seed: int, config_path: str = BASE_CFG, model_override: str = "") -> dict:
    """Zero-array benchmark config for a given slot count."""
    cfg = load_config(config_path)
    if model_override:
        cfg.setdefault("llm", {}).setdefault("huggingface", {})["model"] = model_override
    cfg["environment"]["num_days"]           = 1
    cfg["environment"]["slots_per_day"]      = size
    cfg["environment"]["use_real_calendars"] = False
    cfg["environment"]["availability_rate"]  = 0.0   # all-zero arrays
    cfg["environment"]["intersections"]      = 0
    cfg["simulation"]["seed"]                = seed
    cfg["simulation"]["max_iterations"]      = 1
    cfg["simulation"]["max_planning_rounds"] = 1
    cfg["simulation"]["max_conversation_steps"] = 2
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_one(config: dict, use_ot: bool) -> dict:
    """
    Run a single simulation (OT or plain).
    Returns dict with elapsed, crypto_time, status, error.
    """
    result = {
        "method":      "OT" if use_ot else "PLAIN",
        "size":        config["environment"]["slots_per_day"],
        "seed":        config["simulation"]["seed"],
        "elapsed":     None,
        "crypto_time": 0.0,
        "status":      "error",
        "error":       None,
    }

    original_method = None
    env             = None
    mcp_client      = None

    try:
        import os
        from fastmcp import Client
        from src.communication_protocols.sequential import SequentialCommunicationProtocol
        from src.agents.agent_factory import build_agents
        from src.networks import build_communication_network
        from src.utils import create_environment, get_model_name, get_generation_params
        from src.logger import ToolCallLogger, AgentTrajectoryLogger
        from envs.dcops.meeting_scheduling.meeting_scheduling_env import MeetingSchedulingEnvironment

        config = copy.deepcopy(config)

        # ── Monkey-patch plain AND if needed ──────────────────────────────
        if not use_ot:
            original_method = MeetingSchedulingEnvironment._generate_meeting_intersections

            def _plain_patch(self):
                import time as _t
                self._meeting_intersections_cache = None
                total_slots = self.num_days * self.slots_per_day
                meeting_intersections = {}
                from src.availability import AvailabilityConstants
                for meeting_id, agent_slots in self.meeting_availabilities.items():
                    participants = list(agent_slots.keys())
                    if len(participants) != 2:
                        continue
                    a = agent_slots[participants[0]]
                    b = agent_slots[participants[1]]
                    t0 = _t.time()
                    common = [i for i, (x, y) in enumerate(zip(a, b)) if x == 1 and y == 1]
                    self.crypto_time_s += _t.time() - t0
                    arr = [AvailabilityConstants.BUSY] * total_slots
                    for idx in common:
                        arr[idx] = AvailabilityConstants.AVAILABLE
                    meeting_intersections[meeting_id] = arr
                self._meeting_intersections_cache = meeting_intersections
                return meeting_intersections

            MeetingSchedulingEnvironment._generate_meeting_intersections = _plain_patch

        # ── Setup ─────────────────────────────────────────────────────────
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        config.setdefault("simulation", {})["run_timestamp"] = run_ts

        mcp_url = os.environ.get("MCP_HTTP_URL")
        if mcp_url:
            mcp_client = Client(mcp_url)
        else:
            from src.server import mcp as _srv
            mcp_client = Client(_srv)

        from src.logger import ToolCallLogger, AgentTrajectoryLogger
        tool_logger       = ToolCallLogger("MeetingSchedulingEnvironment", config["simulation"]["seed"], config, run_timestamp=run_ts)
        trajectory_logger = AgentTrajectoryLogger("MeetingSchedulingEnvironment", config["simulation"]["seed"], config, run_timestamp=run_ts)

        comm = SequentialCommunicationProtocol(config, tool_logger, mcp_client, run_timestamp=run_ts)
        env  = create_environment(comm, "MeetingSchedulingEnvironment", config, tool_logger)

        async with mcp_client as client:
            await client.call_tool("initialize_environment_tools",
                                   {"environment_name": "MeetingSchedulingEnvironment"})

        agent_names = env.get_agent_names()
        network     = build_communication_network(agent_names, config)
        env.set_communication_network(network)
        env.tool_logger.reset_log()
        await env.async_init()

        llm_cfg    = config["llm"]
        provider   = llm_cfg.get("provider", "unknown").lower()
        model_name = get_model_name(provider, llm_cfg)
        gen_params = get_generation_params(llm_cfg)
        max_steps  = config["simulation"].get("max_conversation_steps", 2)

        agents = build_agents(
            agent_names,
            provider=provider,
            provider_label=llm_cfg.get("provider", "unknown"),
            llm_config=llm_cfg,
            model_name=model_name,
            max_conversation_steps=max_steps,
            tool_logger=tool_logger,
            trajectory_logger=trajectory_logger,
            environment=env,
            generation_params=gen_params,
            vllm_runtime=None,
        )
        env.set_agent_clients(agents)

        max_iter    = config["simulation"].get("max_iterations", 1)
        max_rounds  = config["simulation"].get("max_planning_rounds", 1)

        # ── Run ───────────────────────────────────────────────────────────
        t_start = time.perf_counter()

        for it in range(1, max_iter + 1):
            if env.done(it):
                break
            for rnd in range(1, max_rounds + 1):
                for agent in env.agents:
                    ctx = env.build_agent_context(
                        agent.name, phase="planning", iteration=it, planning_round=rnd
                    )
                    await comm.agent_planning_turn(agent, agent.name, ctx, env, it, rnd)
            for agent in env.agents:
                ctx = env.build_agent_context(agent.name, phase="execution", iteration=it)
                await comm.agent_execution_turn(agent, agent.name, ctx, env, it)
            env.log_iteration_summary(it)

        env.generate_final_summary()

        result["elapsed"]     = time.perf_counter() - t_start
        result["crypto_time"] = getattr(env, "crypto_time_s", 0.0)
        result["status"]      = "success"

    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"]  = "timeout"
    except Exception as exc:
        result["status"] = "error"
        result["error"]  = f"{type(exc).__name__}: {str(exc)[:120]}"
        logging.warning("run_one failed [%s size=%s]: %s",
                        result["method"], result["size"], result["error"])
    finally:
        if not use_ot and original_method is not None:
            try:
                from envs.dcops.meeting_scheduling.meeting_scheduling_env import MeetingSchedulingEnvironment
                MeetingSchedulingEnvironment._generate_meeting_intersections = original_method
            except Exception:
                pass
        env = None
        mcp_client = None
        gc.collect()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pair runner (OT then Plain, same config)
# ─────────────────────────────────────────────────────────────────────────────

async def run_pair(size: int, seed: int, run_ot: bool,
                   config_path: str = BASE_CFG, model_override: str = "") -> dict:
    base = build_config(size, seed, config_path=config_path, model_override=model_override)

    ot_result    = {"status": "skipped", "elapsed": None, "crypto_time": 0.0}
    plain_result = {"status": "skipped", "elapsed": None, "crypto_time": 0.0}

    if run_ot:
        print(f"    OT   ...", end=" ", flush=True)
        ot_result = await run_one(copy.deepcopy(base), use_ot=True)
        _print_result(ot_result)
        gc.collect()

    print(f"    PLAIN...", end=" ", flush=True)
    plain_result = await run_one(copy.deepcopy(base), use_ot=False)
    _print_result(plain_result)

    overhead    = None
    overhead_pct = None
    if ot_result["elapsed"] and plain_result["elapsed"]:
        ot_cry   = ot_result["crypto_time"]
        pl_cry   = plain_result["crypto_time"]
        overhead = ot_cry - pl_cry
        base_e2e = plain_result["elapsed"]
        overhead_pct = (overhead / base_e2e * 100) if base_e2e > 0 else 0.0

    return {
        "size":         size,
        "seed":         seed,
        "ot_e2e_s":     ot_result["elapsed"],
        "plain_e2e_s":  plain_result["elapsed"],
        "ot_crypto_s":  ot_result["crypto_time"],
        "plain_crypto_s": plain_result["crypto_time"],
        "overhead_s":   overhead,
        "overhead_pct": overhead_pct,
        "ot_status":    ot_result["status"],
        "plain_status": plain_result["status"],
        "ot_error":     ot_result.get("error"),
        "plain_error":  plain_result.get("error"),
    }


def _print_result(r: dict) -> None:
    if r["elapsed"] is not None:
        print(f"✓ {r['elapsed']:.3f}s (crypto: {r['crypto_time']:.6f}s)")
    else:
        print(f"✗ {r['status']}: {r.get('error','')[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(sizes: list[int], num_runs: int, run_ot: bool,
         config_path: str = BASE_CFG, model_override: str = "") -> None:
    configure_logging()
    load_dotenv()
    logging.getLogger().setLevel(logging.WARNING)

    # Resolve effective model name for display
    _cfg_preview = load_config(config_path)
    hf_block = _cfg_preview.get("llm", {}).get("huggingface", {})
    effective_model = (
        model_override
        or hf_block.get("model")
        or _cfg_preview.get("llm", {}).get("model")
        or "unknown"
    )

    print("=" * 80)
    print("SIMULATION BENCHMARK — OT vs PLAIN (zero arrays)")
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {num_runs}  |  OT: {run_ot}")
    print(f"Model: {effective_model}")
    print(f"Config: {config_path}")
    print("=" * 80)

    # ── Warmup: load the LLM model before benchmark starts ───────────────────
    print("\n[warmup] Loading LLM model (not counted in results)...")
    try:
        warmup_cfg = build_config(sizes[0], seed=99, config_path=config_path, model_override=model_override)
        asyncio.run(run_one(copy.deepcopy(warmup_cfg), use_ot=False))
        print("[warmup] Done.\n")
    except Exception as exc:
        print(f"[warmup] Failed (non-fatal): {exc}\n")
    gc.collect()

    all_rows: list[dict] = []

    for size in sizes:
        print(f"\n[size={size}]")
        print("─" * 60)

        for run_idx in range(1, num_runs + 1):
            seed = 100 + run_idx - 1
            print(f"  run {run_idx}/{num_runs} (seed={seed})")
            try:
                row = asyncio.run(run_pair(size, seed, run_ot, config_path, model_override))
                all_rows.append(row)
            except Exception as exc:
                logging.error("Pair failed size=%d run=%d: %s", size, run_idx, exc)
            gc.collect()

        # Per-size summary
        size_rows = [r for r in all_rows if r["size"] == size]
        ok = [r for r in size_rows
              if r["ot_status"] in ("success", "skipped")
              and r["plain_status"] == "success"]
        if ok:
            def _mean(lst): return sum(lst) / len(lst) if lst else 0.0
            # Exclude first run (pyot cold-start outlier) from crypto stats if >2 runs
            ok_excl1 = ok[1:] if len(ok) > 2 else ok
            print(f"\n  Summary ({len(ok)}/{num_runs} ok):")
            if run_ot:
                ot_crypto_vals = [r['ot_crypto_s'] for r in ok_excl1]
                print(f"    OT   e2e : {_mean([r['ot_e2e_s'] for r in ok if r['ot_e2e_s'] is not None]):.3f}s  "
                      f"crypto: {_mean(ot_crypto_vals):.6f}s  (excl. cold-start)")
            plain_crypto_vals = [r['plain_crypto_s'] for r in ok_excl1]
            print(f"    PLAIN e2e: {_mean([r['plain_e2e_s'] for r in ok if r['plain_e2e_s'] is not None]):.3f}s  "
                  f"crypto: {_mean(plain_crypto_vals):.6f}s")
            if run_ot:
                overheads = [r['overhead_s'] for r in ok if r['overhead_s'] is not None]
                overhead_pcts = [r['overhead_pct'] for r in ok if r['overhead_pct'] is not None]
                print(f"    Overhead : {_mean(overheads):.6f}s  ({_mean(overhead_pcts):.4f}%)")

    # ── CSV ──────────────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo results to save.")
        return

    out_dir = PROJECT_ROOT / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"simulation_benchmark_{ts}.csv"

    fields = [
        "size", "seed",
        "ot_e2e_s", "plain_e2e_s",
        "ot_crypto_s", "plain_crypto_s",
        "overhead_s", "overhead_pct",
        "ot_status", "plain_status",
        "ot_error", "plain_error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*80}")
    print(f"Results → {out_csv}")
    print("=" * 80)

    # Final table
    by_size = {}
    for r in all_rows:
        by_size.setdefault(r["size"], []).append(r)

    def _mean(lst): return sum(x for x in lst if x is not None) / max(1, sum(1 for x in lst if x is not None))

    print(f"\n{'Size':>6} | {'OT E2E':>10} | {'Plain E2E':>10} | {'OT Crypto (s)':>14} | {'Plain Crypto (s)':>17} | {'Overhead (s)':>13} | {'N':>4}")
    print("─" * 90)
    for size in sorted(by_size):
        rows = [r for r in by_size[size] if r["plain_status"] == "success"]
        if not rows:
            continue
        # Exclude first run from crypto stats (cold-start outlier)
        rows_excl1 = rows[1:] if len(rows) > 2 else rows
        print(
            f"{size:6d} | "
            f"{_mean([r['ot_e2e_s'] for r in rows]):10.3f} | "
            f"{_mean([r['plain_e2e_s'] for r in rows]):10.3f} | "
            f"{_mean([r['ot_crypto_s'] for r in rows_excl1]):14.6f} | "
            f"{_mean([r['plain_crypto_s'] for r in rows_excl1]):17.6f} | "
            f"{_mean([r['overhead_s'] for r in rows_excl1 if r['overhead_s'] is not None]):13.6f} | "
            f"{len(rows):4d}"
        )
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OT vs Plain simulation benchmark (zero arrays)")
    parser.add_argument("--sizes",  type=int, nargs="+", default=SIZES)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--no-ot",  action="store_true", help="Plain only")
    parser.add_argument("--config", type=str, default=BASE_CFG,
                        help="Path to YAML config (default: meeting_scheduling.yaml)")
    parser.add_argument("--model",  type=str, default="",
                        help=f"Model shortcut or full HF path. Shortcuts: {list(KNOWN_MODELS.keys())}")
    args = parser.parse_args()

    model_override = KNOWN_MODELS.get(args.model, args.model)  # resolve shortcut

    main(args.sizes, args.runs, run_ot=not args.no_ot,
         config_path=args.config, model_override=model_override)
