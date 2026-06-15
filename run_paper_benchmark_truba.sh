#!/bin/bash
#SBATCH --job-name=terrarium-benchmark
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/benchmark_%j.log
#SBATCH --error=logs/benchmark_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# TRUBA Simulation Benchmark — OT vs Plain AND (zero arrays)
# Outputs: tests/results/simulation_benchmark_TIMESTAMP.csv
# Usage:
#   sbatch run_paper_benchmark_truba.sh
#   sbatch run_paper_benchmark_truba.sh --sizes "8 16 32 112" --runs 5
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=================================================="
echo "Terrarium OT vs Plain Simulation Benchmark"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=================================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── Argument parsing ─────────────────────────────────────────────────────────
SIZES="960 480 448 240 224 112 56 32 16 8"
RUNS=5
NO_OT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --runs)  RUNS="$2";  shift 2 ;;
        --no-ot) NO_OT="--no-ot"; shift ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# ── Modules ──────────────────────────────────────────────────────────────────
echo "Loading modules..."
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# ── Virtual environment ───────────────────────────────────────────────────────
echo "Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found (.venv or venv)!"
    exit 1
fi

# ── Environment variables ─────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Load .env if present
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ── Output directories ────────────────────────────────────────────────────────
mkdir -p logs tests/results

# ── Print configuration ───────────────────────────────────────────────────────
echo ""
echo "Configuration:"
echo "  Array sizes : $SIZES"
echo "  Runs/size   : $RUNS"
echo "  OT enabled  : $([ -z '$NO_OT' ] && echo yes || echo no)"
echo "  LLM provider: $(python -c "import yaml; cfg=yaml.safe_load(open('examples/configs/meeting_scheduling.yaml')); print(cfg.get('llm',{}).get('provider','?'))" 2>/dev/null || echo unknown)"
echo ""

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo "Starting benchmark..."

python tests/run_simulation_benchmark.py \
    --sizes $SIZES \
    --runs  $RUNS \
    $NO_OT

# ── Verify results ────────────────────────────────────────────────────────────
echo ""
echo "Checking results..."
RESULTS_FILE=$(ls -1t tests/results/simulation_benchmark_*.csv 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    echo "Results saved: $RESULTS_FILE"
    echo "Preview:"
    head -6 "$RESULTS_FILE"
else
    echo "WARNING: No simulation_benchmark CSV found. Check logs."
fi

echo ""
echo "=================================================="
echo "Completed: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=================================================="
