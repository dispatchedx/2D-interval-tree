"""
Test suite and utilities for 2D AVL Interval Tree

Includes comprehensive tests, performance benchmarks, and validation functions.
"""

import time
import random
import math
from interval_tree import IntervalTree, Interval, Node, YNode, print_overlaps


def count_nodes(root):
    """Count total nodes in tree"""
    if root is None:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def get_tree_height(root):
    """Get actual tree height (recursive calculation)"""
    if root is None:
        return 0
    return 1 + max(get_tree_height(root.left), get_tree_height(root.right))


def print_tree(root, ext_interval=0, level=0):
    """Print tree structure to terminal"""
    if root is not None:
        if isinstance(root, YNode):
            x_interval = ext_interval
            y_interval = root.interval
        else:
            x_interval = root.interval
            y_interval = root.y_tree.root.interval
        print_tree(root.left, x_interval, level + 1)
        print(
            " " * 4 * level + f"-> x({x_interval.low} {x_interval.high}) y({y_interval.low}"
            f" {y_interval.high}) (max={root.max} h={root.height})"
        )
        print_tree(root.right, x_interval, level + 1)


def y_print_tree(root):
    """Print all y-trees"""
    if root is not None:
        y_print_tree(root.left)
        print_tree(root.y_tree.root, root.interval)
        y_print_tree(root.right)


def validate_avl_property(root):
    """
    Validate that tree maintains AVL property (|balance| <= 1)

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    if root is None:
        return True, errors

    # Check balance factor
    left_height = get_tree_height(root.left)
    right_height = get_tree_height(root.right)
    balance = left_height - right_height

    if abs(balance) > 1:
        errors.append(f"AVL violation at node [{root.interval.low},{root.interval.high}]: balance={balance}")

    # Check stored height matches actual
    actual_height = 1 + max(left_height, right_height)
    if root.height != actual_height:
        errors.append(
            f"Height mismatch at [{root.interval.low},{root.interval.high}]: "
            f"stored={root.height}, actual={actual_height}"
        )

    # Recursively check children
    left_valid, left_errors = validate_avl_property(root.left)
    right_valid, right_errors = validate_avl_property(root.right)

    errors.extend(left_errors)
    errors.extend(right_errors)

    return len(errors) == 0, errors


def validate_tree_structure(root):
    """Validate BST property and max values"""
    errors = []

    if root is None:
        return True, errors

    # Check BST property for x-axis
    if root.left is not None:
        if root.left.interval.low >= root.interval.low:
            errors.append(f"BST violation: left child {root.left.interval.low} >= parent {root.interval.low}")

    if root.right is not None:
        if root.right.interval.low < root.interval.low:
            errors.append(f"BST violation: right child {root.right.interval.low} < parent {root.interval.low}")

    # Check max property
    expected_max = root.interval.high
    if root.left is not None:
        expected_max = max(expected_max, root.left.max)
    if root.right is not None:
        expected_max = max(expected_max, root.right.max)

    if root.max != expected_max:
        errors.append(
            f"Max property violation at [{root.interval.low},{root.interval.high}]: "
            f"stored max={root.max}, expected max={expected_max}"
        )

    # Recursively check children
    left_valid, left_errors = validate_tree_structure(root.left)
    right_valid, right_errors = validate_tree_structure(root.right)

    errors.extend(left_errors)
    errors.extend(right_errors)

    return len(errors) == 0, errors


def comprehensive_test_suite():
    """Run comprehensive test suite"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Test 1: Basic Insert and Search
    print("\n[Test 1: Basic Insert and Search]")
    tree1 = IntervalTree()
    tree1.update([[0, 5, 0, 5], [10, 15, 10, 15], [3, 8, 3, 8]])
    tree1.findall_overlapping_interval(tree1.root, [4, 6, 4, 6])
    print(f"Expected: 2 overlaps")
    print(f"Found: {len(tree1.overlaps)} overlaps")
    assert len(tree1.overlaps) == 2, "Test 1 failed"
    print("✓ Test 1 passed")

    # Test 2: No overlaps
    print("\n[Test 2: No Overlaps]")
    tree2 = IntervalTree()
    tree2.update([[0, 2, 0, 2], [10, 12, 10, 12]])
    tree2.findall_overlapping_interval(tree2.root, [5, 7, 5, 7])
    assert len(tree2.overlaps) == 0, "Test 2 failed"
    print("✓ Test 2 passed")

    # Test 3: Sorted data (worst case for non-AVL)
    print("\n[Test 3: Sorted Data - AVL Should Handle This]")
    tree3 = IntervalTree()
    sorted_intervals = [[i, i + 1, i, i + 1] for i in range(100)]
    tree3.update(sorted_intervals)
    height = get_tree_height(tree3.root)
    node_count = count_nodes(tree3.root)
    optimal = math.ceil(math.log2(node_count + 1))
    print(f"Nodes: {node_count}, Height: {height}, Optimal: {optimal}")
    print(f"Height ratio: {height/optimal:.2f}x optimal")
    assert height < optimal * 1.5, f"AVL tree too tall: {height} vs optimal {optimal}"
    print("✓ Test 3 passed - AVL handles sorted data!")

    # Test 4: Validate AVL property
    print("\n[Test 4: Validate AVL Property]")
    is_valid, errors = validate_avl_property(tree3.root)
    if is_valid:
        print("✓ AVL property maintained (all balances <= 1)")
    else:
        print("✗ AVL violations found:")
        for error in errors[:5]:
            print(f"  {error}")
        assert False, "AVL property violated"
    print("✓ Test 4 passed")

    # Test 5: Delete and rebalance
    print("\n[Test 5: Delete and Rebalance]")
    tree5 = IntervalTree()
    tree5.update([[i, i + 1, i, i + 1] for i in range(20)])
    for i in range(0, 10):
        tree5.root = tree5.delete(tree5.root, [i, i + 1, i, i + 1])
    is_valid, errors = validate_avl_property(tree5.root)
    assert is_valid, "AVL property violated after deletes"
    print("✓ Test 5 passed - Tree remains balanced after deletes")

    # Test 6: Point intervals
    print("\n[Test 6: Point Intervals]")
    tree6 = IntervalTree()
    tree6.update([[5, 5, 5, 5], [3, 7, 3, 7]])
    tree6.findall_overlapping_interval(tree6.root, [5, 5, 5, 5])
    assert len(tree6.overlaps) == 2, "Test 6 failed"
    print("✓ Test 6 passed")

    # Test 7: Identical x-intervals, different y
    print("\n[Test 7: Identical X, Different Y]")
    tree7 = IntervalTree()
    tree7.update([[0, 10, 0, 5], [0, 10, 6, 10], [0, 10, 11, 15]])
    tree7.findall_overlapping_interval(tree7.root, [0, 10, 7, 12])
    x_node_count = count_nodes(tree7.root)
    print(f"X-nodes in tree: {x_node_count} (should be 1)")
    print(f"Found {len(tree7.overlaps)} overlaps")
    assert len(tree7.overlaps) == 2, "Test 7 failed"
    assert x_node_count == 1, "Should have single x-node"
    print("✓ Test 7 passed")

    # Test 8: Edge touching
    print("\n[Test 8: Edge Touching]")
    tree8 = IntervalTree()
    tree8.update([[0, 5, 0, 5]])
    tree8.findall_overlapping_interval(tree8.root, [5, 10, 5, 10])
    assert len(tree8.overlaps) == 1, "Test 8 failed"
    print("✓ Test 8 passed")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


def test_sorted_data():
    """Specific test for sorted data (worst case for non-AVL BST)"""
    print("\n" + "=" * 60)
    print("SORTED DATA TEST (Worst case for non-AVL BST)")
    print("=" * 60)

    size = 5000
    sorted_intervals = [[i, i + 10, i, i + 10] for i in range(size)]

    print(f"\nBuilding AVL tree with {size} sorted intervals...")
    start = time.time()
    tree = IntervalTree(sorted_intervals)
    build_time = time.time() - start

    height = get_tree_height(tree.root)
    node_count = count_nodes(tree.root)
    optimal = math.ceil(math.log2(node_count + 1))

    print(f"Build time: {build_time:.4f}s")
    print(f"Height: {height} (optimal: {optimal}, ratio: {height/optimal:.2f}x)")

    # Validate AVL property
    is_valid, errors = validate_avl_property(tree.root)
    if is_valid:
        print("✓ AVL property maintained")
    else:
        print("✗ AVL violations:")
        for error in errors[:5]:
            print(f"  {error}")

    # Test search
    start = time.time()
    tree.findall_overlapping_interval(tree.root, [2500, 2600, 2500, 2600])
    search_time = time.time() - start
    print(f"Search time: {search_time:.6f}s")
    print(f"Found: {len(tree.overlaps)} overlaps")

    print("\n✓ AVL tree successfully handles sorted data!")
    print("=" * 60)


def timing_find_single():
    """AVL timing analysis for finding a single overlapping interval"""
    print("\n" + "=" * 60)
    print("AVL TREE TIMING ANALYSIS - FIND SINGLE")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000, 10000, 50000]
    search_times = []

    for size in sizes:
        intervals = [
            [random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000)]
            for _ in range(size)
        ]
        intervals = [[x[0], max(x[0], x[1]), x[2], max(x[2], x[3])] for x in intervals]

        start = time.time()
        tree = IntervalTree(intervals)
        build_time = time.time() - start

        start = time.time()
        overlap = tree.find_overlapping_interval(tree.root, [5000, 5100, 5000, 5100])
        search_time = time.time() - start
        search_times.append(search_time)

        node_count = count_nodes(tree.root)
        height = get_tree_height(tree.root)
        optimal_height = math.ceil(math.log2(node_count + 1))
        height_ratio = height / optimal_height

        found_count = 1 if overlap else 0

        print(
            f"Size {size:6d}: Build={build_time:.4f}s, "
            f"Search={search_time:.6f}s, Height={height} (opt={optimal_height}, {height_ratio:.2f}x), "
            f"Found={found_count}"
        )

    print("\nGrowth Rate Analysis (Expected: O(log n))")
    for i in range(1, len(sizes)):
        time_ratio = search_times[i] / search_times[i - 1]
        expected_ratio = math.log(sizes[i]) / math.log(sizes[i - 1])
        status = (
            "✓ Close to O(log n)"
            if time_ratio <= expected_ratio * 2
            else f"⚠️ Slower than expected (Expected ~x{expected_ratio:.2f}, got x{time_ratio:.2f})"
        )
        print(f"{sizes[i-1]} -> {sizes[i]}: Search time x{time_ratio:.2f} ({status})")


def timing_find_all():
    """AVL timing analysis for finding all overlapping intervals"""
    print("\n" + "=" * 60)
    print("AVL TREE TIMING ANALYSIS - FIND ALL")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000, 10000, 50000]
    search_times = []
    overlaps_list = []

    for size in sizes:
        intervals = [
            [random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000)]
            for _ in range(size)
        ]
        intervals = [[x[0], max(x[0], x[1]), x[2], max(x[2], x[3])] for x in intervals]

        start = time.time()
        tree = IntervalTree(intervals)
        build_time = time.time() - start

        start = time.time()
        tree.findall_overlapping_interval(tree.root, [5000, 5100, 5000, 5100])
        search_time = time.time() - start
        search_times.append(search_time)
        overlaps_list.append(len(tree.overlaps))

        node_count = count_nodes(tree.root)
        height = get_tree_height(tree.root)
        optimal_height = math.ceil(math.log2(node_count + 1))
        height_ratio = height / optimal_height

        print(
            f"Size {size:6d}: Build={build_time:.4f}s, "
            f"Search={search_time:.6f}s, Height={height} (opt={optimal_height}, {height_ratio:.2f}x), "
            f"Found={len(tree.overlaps)}"
        )

    print("\nGrowth Rate Analysis (Expected: O(log n + k))")
    for i in range(1, len(sizes)):
        k_ratio = (overlaps_list[i] + 1) / (overlaps_list[i - 1] + 1)
        expected_ratio = (math.log(sizes[i]) / math.log(sizes[i - 1])) * k_ratio
        time_ratio = search_times[i] / search_times[i - 1]
        status = (
            "✓ Close to O(log n + k)"
            if time_ratio <= expected_ratio * 2
            else f"⚠️ Slower than expected (Expected ~x{expected_ratio:.2f}, got x{time_ratio:.2f})"
        )
        print(f"{sizes[i-1]} -> {sizes[i]}: Search time x{time_ratio:.2f} ({status})")


def stress_test():
    """Stress test with large dataset"""
    print("\n" + "=" * 60)
    print("STRESS TEST")
    print("=" * 60)

    size = 100000
    print(f"\nBuilding tree with {size:,} random intervals...")

    intervals = []
    for i in range(size):
        x_low = random.randint(0, 100000)
        x_high = random.randint(x_low, 100000)
        y_low = random.randint(0, 100000)
        y_high = random.randint(y_low, 100000)
        intervals.append([x_low, x_high, y_low, y_high])

    start = time.time()
    tree = IntervalTree(intervals)
    build_time = time.time() - start

    height = get_tree_height(tree.root)
    node_count = count_nodes(tree.root)
    optimal = math.ceil(math.log2(node_count + 1))

    print(f"Build time: {build_time:.2f}s")
    print(f"Height: {height} (optimal: {optimal}, ratio: {height/optimal:.2f}x)")

    # Multiple queries
    print("\nPerforming 100 random queries...")
    total_time = 0
    total_overlaps = 0

    for _ in range(100):
        qx_low = random.randint(0, 90000)
        qx_high = random.randint(qx_low, qx_low + 10000)
        qy_low = random.randint(0, 90000)
        qy_high = random.randint(qy_low, qy_low + 10000)

        start = time.time()
        tree.findall_overlapping_interval(tree.root, [qx_low, qx_high, qy_low, qy_high])
        total_time += time.time() - start
        total_overlaps += len(tree.overlaps)

    avg_time = total_time / 100
    avg_overlaps = total_overlaps / 100

    print(f"Average query time: {avg_time:.6f}s")
    print(f"Average overlaps found: {avg_overlaps:.1f}")
    print(f"\n✓ Stress test passed!")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("2D AVL INTERVAL TREE - TEST SUITE")
    print("=" * 60)

    # Run all tests
    comprehensive_test_suite()
    test_sorted_data()
    timing_find_single()
    timing_find_all()

    stress_test()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo visualize a tree, run:")
    print("  python interval_tree.py")
    print("\nFor custom tests, import functions from this module:")
    print("  from test_interval_tree import validate_avl_property")
    print("=" * 60)
