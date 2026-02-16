"""DCOP environments module - contains MeetingScheduling.

This package depends on the external CoLLAB submodule for instance generation
and scoring. CoLLAB is not a Python package at its root, so we add
`external/CoLLAB` to sys.path to make `problem_layer.*` imports available.
"""

from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

_COLLAB_ROOT = Path(__file__).resolve().parents[2] / "external" / "CoLLAB"
if _COLLAB_ROOT.exists():
    collab_str = str(_COLLAB_ROOT)
    if collab_str not in sys.path:
        sys.path.insert(0, collab_str)

# Import MeetingScheduling (simplified, no CoLLAB dependency)
from .meeting_scheduling import MeetingSchedulingEnvironment

__all__ = ['MeetingSchedulingEnvironment']
