# Quick run wrapper for Windows (Local development)
# Usage: .\run.ps1

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found. Run: python -m venv .venv; pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

# Run simulation
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml $args
