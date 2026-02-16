"""
Cryptographic protocols for privacy-preserving multi-agent coordination.

This module provides Oblivious Transfer (OT) protocol implementation
for secure slot intersection in meeting scheduling scenarios.
"""

from .ot_manager import OTManager, compute_private_intersection, OT_AVAILABLE

__all__ = ['OTManager', 'compute_private_intersection', 'OT_AVAILABLE']
