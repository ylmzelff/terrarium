#!/bin/bash
# Quick run wrapper for TRUBA HPC
# Usage: ./run.sh

# Load modules (TRUBA-specific)
module load centos7.9/comp/python/3.11 2>/dev/null
module load centos7.9/lib/gmp/6.2.1 2>/dev/null

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup_truba.sh first!"
    exit 1
fi

# Run simulation
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml "$@"
