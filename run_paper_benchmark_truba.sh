#!/bin/bash
#SBATCH --job-name=terrarium-benchmark
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

cd /arf/scratch/egitimg15u2/terrarium

source terrarium_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/arf/scratch/egitimg15u2/terrarium:/arf/scratch/egitimg15u2/terrarium/crypto:$PYTHONPATH
export PYTHONUNBUFFERED=1

# ── Argument parsing ──────────────────────────────────────────────────────────
SIZES="960 480 448 240 224 112 56 32 16 8"
RUNS=5
NO_OT=""
MODEL=""   # e.g. qwen | llama | mistral | or full HF path

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)  SIZES="$2";  shift 2 ;;
        --runs)   RUNS="$2";   shift 2 ;;
        --no-ot)  NO_OT="--no-ot"; shift ;;
        --model)  MODEL="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

MODEL_ARG=""
if [ -n "$MODEL" ]; then
    MODEL_ARG="--model $MODEL"
fi

# ── Build crypto extension ────────────────────────────────────────────────────
cd crypto
python3 setup.py build_ext --inplace
cd ..

# ── Output dirs ───────────────────────────────────────────────────────────────
mkdir -p logs tests/results

echo "=================================================="
echo "Terrarium OT vs Plain Simulation Benchmark"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Sizes: $SIZES | Runs: $RUNS"
echo "=================================================="

# ── Run benchmark ─────────────────────────────────────────────────────────────
python3 tests/run_simulation_benchmark.py \
    --sizes $SIZES \
    --runs  $RUNS \
    $NO_OT \
    $MODEL_ARG

# ── Verify results ────────────────────────────────────────────────────────────
RESULTS_FILE=$(ls -1t tests/results/simulation_benchmark_*.csv 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    echo "Results saved: $RESULTS_FILE"
    head -6 "$RESULTS_FILE"
else
    echo "WARNING: No results CSV found. Check logs/benchmark_${SLURM_JOB_ID}.err"
fi

echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
