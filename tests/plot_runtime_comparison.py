"""
Runtime Comparison Plot: Baseline vs OT-128 vs OT-256
======================================================
Generates a log-scale line chart similar to the paper figure.

Usage:
    python tests/plot_runtime_comparison.py
"""

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from matplotlib.transforms import ScaledTranslation
except ImportError:
    print("Install dependencies: pip install matplotlib numpy")
    sys.exit(1)

# --- Data (seconds) ---
slot_sizes = [8, 16, 32, 56, 112, 224, 240, 448, 480, 960]

baseline = [1.0e-6, 2.1e-6, 3.1e-6, 4.0e-6, 7.1e-6,
            1.4e-5, 1.5e-5, 3.0e-5, 3.1e-5, 6.7e-5]

ot_128   = [4.0e-4, 4.5e-4, 5.5e-4, 8.5e-4, 1.9e-3,
            5.5e-3, 6.4e-3, 2.0e-2, 2.3e-2, 9.0e-2]

ot_256   = [4.1e-4, 4.8e-4, 6.2e-4, 9.7e-4, 2.3e-3,
            7.3e-3, 8.3e-3, 2.7e-2, 3.1e-2, 1.4e-1]

# Keep data in seconds
baseline_ms = baseline
ot_128_ms   = ot_128
ot_256_ms   = ot_256

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(slot_sizes, baseline_ms,
        marker='o', linewidth=2.2, markersize=8,
        label='Baseline', color='#1f77b4')

ax.plot(slot_sizes, ot_128_ms,
        marker='s', linewidth=2.2, markersize=8,
        label='OT-128', color='#ff7f0e')

ax.plot(slot_sizes, ot_256_ms,
        marker='^', linewidth=2.2, markersize=9,
        label='OT-256', color='#2ca02c')

# Log scale on y-axis
ax.set_yscale('log')

# X-axis ticks at exact slot sizes
ax.set_xticks(slot_sizes)
ax.set_xticklabels([str(s) for s in slot_sizes], rotation=45, fontsize=22,
                   ha='right', va='top')
# Fine-tune individual labels to avoid overlap
shifts = { 1: 12, 2: 22, 3: 32,4: 25, 6: 20, 8: 20, 9: 20}  # index: points (negative=left, positive=right)
for idx, pts in shifts.items():
    lbl = ax.get_xticklabels()[idx]
    lbl.set_transform(lbl.get_transform() + ScaledTranslation(pts/72, 0, fig.dpi_scale_trans))
ax.tick_params(axis='y', labelsize=24)

ax.set_xlabel('Number of Slots', fontsize=28, labelpad=12)
ax.set_ylabel('Runtime (seconds)', fontsize=28, labelpad=12)
ax.set_title('Runtime Comparison vs Slot Size', fontsize=28, pad=16)

ax.legend(fontsize=18, framealpha=0.9)
ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()

output_path = Path(__file__).parent / "results" / "runtime_comparison.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()
