#!/bin/bash
# TRUBA HPC Cluster Setup Script for Terrarium
# Run this script after uploading the project to TRUBA

set -e  # Exit on error

echo "=========================================="
echo "Terrarium Setup for TRUBA HPC"
echo "=========================================="

# 1. Load required modules (TRUBA-specific)
echo ""
echo "[1/5] Loading TRUBA modules..."
module purge
module load centos7.9/comp/python/3.11
module load centos7.9/lib/gmp/6.2.1  # GMP for crypto module
module load centos7.9/comp/gcc/11

# 2. Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 4. Install Python dependencies
echo ""
echo "[4/5] Installing Python packages..."
pip install -r requirements.txt

# 5. Build crypto module (OT protocol)
echo ""
echo "[5/5] Building crypto module..."
cd crypto
python setup.py install
cd ..

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the simulation:"
echo "  1. Activate venv:  source .venv/bin/activate"
echo "  2. Run command:    python examples/base_main.py --config examples/configs/meeting_scheduling.yaml"
echo ""
echo "For SLURM job submission, use: sbatch run_simulation.slurm"
echo ""
