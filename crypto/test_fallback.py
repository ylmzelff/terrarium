#!/usr/bin/env python3
"""
Test script for slot intersection without OT module.
Demonstrates fallback mode functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_blackboard_fallback():
    """Test blackboard integration with fallback mode (no OT)."""
    print("=" * 60)
    print("BLACKBOARD SLOT INTERSECTION TEST (Fallback Mode)")
    print("=" * 60)
    
    try:
        from src.blackboard import Megaboard
        
        megaboard = Megaboard()
        
        # Test case from blackboard log
        agent_a_slots = [2, 3, 4, 6, 7, 8]
        agent_b_slots = [1, 3, 5, 6, 7, 10]
        
        print(f"\nAgent A availability: {agent_a_slots}")
        print(f"Agent B availability: {agent_b_slots}")
        
        # This will use fallback (simple intersection) since OT not built
        result = megaboard.compute_private_intersection(agent_a_slots, agent_b_slots)
        
        print(f"\nCommon slots: {result}")
        
        expected = [3, 6, 7]
        if result == expected:
            print("\n✓ Test PASSED: Blackboard integration works")
            print("  (Using fallback mode - OT module not available)")
            return True
        else:
            print(f"\n✗ Test FAILED: Expected {expected}, got {result}")
            return False
            
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases_fallback():
    """Test edge cases with fallback mode."""
    print("\n" + "=" * 60)
    print("EDGE CASES TEST (Fallback Mode)")
    print("=" * 60)
    
    try:
        from src.blackboard import Megaboard
        
        megaboard = Megaboard()
        
        # Case 1: No intersection
        print("\nCase 1: No common slots")
        a = [0, 2, 4]
        b = [1, 3, 5]
        result = megaboard.compute_private_intersection(a, b)
        print(f"  A: {a}, B: {b} → Result: {result}")
        assert result == [], f"Expected [], got {result}"
        print("  ✓ PASSED")
        
        # Case 2: Full intersection
        print("\nCase 2: All slots common")
        a = [0, 1, 2]
        b = [0, 1, 2]
        result = megaboard.compute_private_intersection(a, b)
        print(f"  A: {a}, B: {b} → Result: {result}")
        assert result == [0, 1, 2], f"Expected [0, 1, 2], got {result}"
        print("  ✓ PASSED")
        
        # Case 3: Single common slot
        print("\nCase 3: Single common slot")
        a = [0, 2, 4, 6]
        b = [1, 3, 4, 5]
        result = megaboard.compute_private_intersection(a, b)
        print(f"  A: {a}, B: {b} → Result: {result}")
        assert result == [4], f"Expected [4], got {result}"
        print("  ✓ PASSED")
        
        print("\n✓ All edge cases PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Edge case FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run fallback mode tests."""
    print("\n" + "=" * 60)
    print("SLOT INTERSECTION TEST - FALLBACK MODE")
    print("(OT module not built - using simple set intersection)")
    print("=" * 60)
    
    results = []
    
    results.append(("Blackboard Integration (Fallback)", test_blackboard_fallback()))
    results.append(("Edge Cases (Fallback)", test_edge_cases_fallback()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:40} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All fallback tests PASSED!")
        print("\nNOTE: To enable privacy-preserving OT:")
        print("  1. Install GMP: conda install -c conda-forge gmp")
        print("  2. Build module: cd crypto && python setup.py install")
        print("  3. Run full tests: python test_ot.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
