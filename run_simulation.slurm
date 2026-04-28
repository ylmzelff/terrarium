#!/bin/bash
#SBATCH --job-name=terrarium_meeting
#SBATCH --account=YOUR_ACCOUNT_ID     # TRUBA hesap ID'nizi buraya yazın
#SBATCH --partition=akya-cuda         # GPU partition (akya-cuda, barbun-cuda, etc.)
#SBATCH --qos=cuda                    # QoS: cuda veya normal
#SBATCH --gres=gpu:1                  # 1 GPU talep et (vLLM için)
#SBATCH --nodes=1                     # Tek node
#SBATCH --ntasks-per-node=1           # Node başına 1 task
#SBATCH --cpus-per-task=8             # 8 CPU core
#SBATCH --mem=32G                     # 32 GB RAM
#SBATCH --time=02:00:00               # Max 2 saat (format: HH:MM:SS)
#SBATCH --output=logs/slurm-%j.out    # Çıktı dosyası (%j = job ID)
#SBATCH --error=logs/slurm-%j.err     # Hata dosyası

# ========================================
# TRUBA Terrarium Simulation Job
# ========================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

# 1. Load modules
module purge
module load centos7.9/comp/python/3.11
module load centos7.9/lib/gmp/6.2.1
module load centos7.9/comp/gcc/11
module load centos7.9/comp/cuda/11.8  # CUDA for GPU

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Create logs directory if not exists
mkdir -p logs

# 4. Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# 5. Run simulation
echo "Starting Terrarium simulation..."
echo ""

python examples/base_main.py \
    --config examples/configs/meeting_scheduling.yaml \
    --note "TRUBA_run_$(date +%Y%m%d_%H%M%S)"

# 6. Job completion
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
