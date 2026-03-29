"""
test_plain_intersection.py
==========================
Unit tests for tests/plain_intersection.py

Run with:
    cd c:/Users/lenovo/Terrarium
    python -m pytest tests/test_plain_intersection.py -v
"""

import sys
from pathlib import Path

# Make sure the tests package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from plain_intersection import PlainIntersectionManager, plain_intersection


# ---------------------------------------------------------------------------
# plain_intersection() function tests
# ---------------------------------------------------------------------------

class TestPlainIntersection:

    def test_empty_arrays(self):
        assert plain_intersection([], []) == []

    def test_no_overlap(self):
        assert plain_intersection([1, 0, 1], [0, 1, 0]) == []

    def test_full_overlap(self):
        a = [1, 1, 1, 1]
        b = [1, 1, 1, 1]
        assert plain_intersection(a, b) == [0, 1, 2, 3]

    def test_partial_overlap(self):
        a = [0, 1, 1, 0, 1]
        b = [1, 1, 0, 0, 1]
        assert plain_intersection(a, b) == [1, 4]

    def test_all_zeros(self):
        assert plain_intersection([0, 0, 0], [0, 0, 0]) == []

    def test_single_slot_match(self):
        assert plain_intersection([1], [1]) == [0]

    def test_single_slot_no_match(self):
        assert plain_intersection([1], [0]) == []

    def test_result_is_sorted(self):
        # Construct a case where the first matching slot is at the end
        a = [0, 1, 0, 1, 0, 1]
        b = [0, 1, 0, 1, 0, 1]
        result = plain_intersection(a, b)
        assert result == sorted(result)

    def test_large_array(self):
        """32 slots, first half available for A, second half for B → no overlap."""
        a = [1] * 16 + [0] * 16
        b = [0] * 16 + [1] * 16
        assert plain_intersection(a, b) == []

    def test_large_array_with_overlap(self):
        n = 32
        a = [i % 2 for i in range(n)]          # 1,0,1,0,...  even indices busy
        b = [1] * n                              # all available
        expected = [i for i in range(n) if i % 2 == 1]
        assert plain_intersection(a, b) == expected

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            plain_intersection([1, 0], [1, 0, 1])

    def test_non_binary_value_raises(self):
        with pytest.raises(ValueError, match="binary"):
            plain_intersection([0, 2, 1], [1, 1, 1])


# ---------------------------------------------------------------------------
# PlainIntersectionManager tests
# ---------------------------------------------------------------------------

class TestPlainIntersectionManager:

    def setup_method(self):
        self.mgr = PlainIntersectionManager()

    def test_basic(self):
        a = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
        b = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]
        result = self.mgr.compute_intersection(a, b)
        assert result == [3, 6, 7]

    def test_total_slots_param_ignored_gracefully(self):
        """total_slots is accepted for API-parity but doesn't affect the result."""
        a = [1, 0, 1]
        b = [1, 1, 0]
        r1 = self.mgr.compute_intersection(a, b)
        r2 = self.mgr.compute_intersection(a, b, total_slots=99)
        assert r1 == r2

    def test_bit_size_param_accepted(self):
        """bit_size is accepted for API-parity."""
        mgr128 = PlainIntersectionManager(bit_size=128)
        mgr256 = PlainIntersectionManager(bit_size=256)
        a = [1, 0, 1, 1]
        b = [1, 1, 0, 1]
        assert mgr128.compute_intersection(a, b) == mgr256.compute_intersection(a, b)

    def test_known_csv_row_config1(self):
        """Config 1 from CSV: both all-zeros → empty intersection."""
        a = [0] * 32
        b = [0] * 32
        assert self.mgr.compute_intersection(a, b) == []

    def test_known_csv_row_config6(self):
        """Config 6: intersection_density=0.1, actual_intersections=3."""
        a = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = self.mgr.compute_intersection(a, b)
        assert len(result) == 3
        assert result == [8, 19, 20]
