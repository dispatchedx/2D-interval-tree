"""
Test suite and utilities for 2D AVL Interval Tree
"""

import time
import random
import math
from interval_tree import IntervalTree, Interval, Node, YNode


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
    overlaps = tree1.findall_overlapping(tree1.root, [4, 6, 4, 6])
    assert len(overlaps) == 2, "Test 1 failed"
    print(f"✓ Found {len(overlaps)} overlaps (expected 2)")

    # Test 2: No overlaps
    print("\n[Test 2: No Overlaps]")
    tree2 = IntervalTree()
    tree2.update([[0, 2, 0, 2], [10, 12, 10, 12]])
    overlaps = tree2.findall_overlapping(tree2.root, [5, 7, 5, 7])
    assert len(overlaps) == 0, "Test 2 failed"
    print("✓ Correctly found no overlaps")

    # Test 3: Sorted data (worst case for non-AVL)
    print("\n[Test 3: Sorted Data - AVL Balancing]")
    tree3 = IntervalTree()
    sorted_intervals = [[i, i + 1, i, i + 1] for i in range(100)]
    tree3.update(sorted_intervals)
    height = get_tree_height(tree3.root)
    node_count = count_nodes(tree3.root)
    optimal = math.ceil(math.log2(node_count + 1))
    print(f"Nodes: {node_count}, Height: {height}, Optimal: {optimal} ({height/optimal:.2f}x)")
    assert height < optimal * 1.5, f"Tree too tall: {height} vs optimal {optimal}"
    print("✓ AVL handles sorted data efficiently")

    # Test 4: Validate AVL property
    print("\n[Test 4: Validate AVL Property]")
    is_valid, errors = validate_avl_property(tree3.root)
    assert is_valid, f"AVL property violated: {errors}"
    print("✓ AVL property maintained")

    # Test 5: Delete and rebalance
    print("\n[Test 5: Delete and Rebalance]")
    tree5 = IntervalTree()
    tree5.update([[i, i + 1, i, i + 1] for i in range(20)])
    for i in range(0, 10):
        tree5.root = tree5.delete(tree5.root, [i, i + 1, i, i + 1])
    is_valid, errors = validate_avl_property(tree5.root)
    assert is_valid, "AVL property violated after deletes"
    print("✓ Tree remains balanced after deletions")

    # Test 6: Point intervals
    print("\n[Test 6: Point Intervals]")
    tree6 = IntervalTree()
    tree6.update([[5, 5, 5, 5], [3, 7, 3, 7]])
    overlaps = tree6.findall_overlapping(tree6.root, [5, 5, 5, 5])
    assert len(overlaps) == 2, "Test 6 failed"
    print(f"✓ Point intervals work correctly ({len(overlaps)} overlaps)")

    # Test 7: Identical x-intervals, different y
    print("\n[Test 7: Identical X, Different Y]")
    tree7 = IntervalTree()
    tree7.update([[0, 10, 0, 5], [0, 10, 6, 10], [0, 10, 11, 15]])
    overlaps = tree7.findall_overlapping(tree7.root, [0, 10, 7, 12])
    x_node_count = count_nodes(tree7.root)
    assert len(overlaps) == 2, "Test 7 failed"
    assert x_node_count == 1, "Should have single x-node"
    print(f"✓ Y-tree structure works ({x_node_count} x-node, {len(overlaps)} overlaps)")

    # Test 8: Edge touching
    print("\n[Test 8: Edge Touching]")
    tree8 = IntervalTree()
    tree8.update([[0, 5, 0, 5]])
    overlaps = tree8.findall_overlapping(tree8.root, [5, 10, 5, 10])
    assert len(overlaps) == 1, "Test 8 failed"
    print("✓ Edge-touching intervals detected")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


def performance_benchmark():
    """Performance benchmark with timing analysis"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000, 10000, 50000]

    print("\n[Find Single Overlap - Expected O(log n)]")
    print(f"{'Size':<8} {'Query(s)':<12} {'Height':<10} {'Optimal':<10} {'Status':<45}")
    print("-" * 85)

    single_times = []
    for size in sizes:
        # Generate random intervals
        intervals = []
        for _ in range(size):
            x_low = random.randint(0, 10000)
            x_high = random.randint(x_low, x_low + 1000)
            y_low = random.randint(0, 10000)
            y_high = random.randint(y_low, y_low + 1000)
            intervals.append([x_low, x_high, y_low, y_high])

        tree = IntervalTree(intervals)
        node_count = count_nodes(tree.root)
        height = get_tree_height(tree.root)
        optimal = math.ceil(math.log2(node_count + 1))
        height_ratio = height / optimal

        # Query tree for single overlap
        start = time.time()
        overlap = tree.find_overlapping(tree.root, [5000, 5100, 5000, 5100])
        query_time = time.time() - start
        single_times.append(query_time)

        # Check if growth rate is logarithmic
        if len(single_times) > 1:
            time_ratio = single_times[-1] / single_times[-2]
            size_ratio = sizes[len(single_times) - 1] / sizes[len(single_times) - 2]
            expected_ratio = math.log(sizes[len(single_times) - 1]) / math.log(sizes[len(single_times) - 2])

            if time_ratio <= expected_ratio * 2:
                status = f"✓ O(log n) ({time_ratio:.2f}x, exp ~{expected_ratio:.2f}x)"
            else:
                status = f"⚠️  Slow ({time_ratio:.2f}x, exp ~{expected_ratio:.2f}x)"
        else:
            status = "Baseline"

        height_str = f"{height} ({height_ratio:.2f}x)"
        print(f"{size:<8} {query_time:<12.6f} {height_str:<10} {optimal:<10} {status:<45}")

    print("\n[Find All Overlaps - Expected O(log n + k)]")
    print(f"{'Size':<8} {'Query(s)':<12} {'Found(k)':<10} {'Height':<10} {'Optimal':<10} {'Status':<35}")
    print("-" * 95)

    all_times = []
    all_found = []
    for size in sizes:
        # Generate random intervals
        intervals = []
        for _ in range(size):
            x_low = random.randint(0, 10000)
            x_high = random.randint(x_low, x_low + 1000)
            y_low = random.randint(0, 10000)
            y_high = random.randint(y_low, y_low + 1000)
            intervals.append([x_low, x_high, y_low, y_high])

        tree = IntervalTree(intervals)
        node_count = count_nodes(tree.root)
        height = get_tree_height(tree.root)
        optimal = math.ceil(math.log2(node_count + 1))
        height_ratio = height / optimal

        # Query tree for all overlaps
        start = time.time()
        overlaps = tree.findall_overlapping(tree.root, [5000, 5100, 5000, 5100])
        query_time = time.time() - start
        all_times.append(query_time)
        all_found.append(len(overlaps))

        # Check if growth rate is O(log n + k)
        if len(all_times) > 1:
            time_ratio = all_times[-1] / all_times[-2]
            k_ratio = (all_found[-1] + 1) / (all_found[-2] + 1)  # +1 to avoid division issues
            expected_ratio = (math.log(sizes[len(all_times) - 1]) / math.log(sizes[len(all_times) - 2])) * k_ratio

            if time_ratio <= expected_ratio * 2:
                status = f"✓ O(log n+k) ({time_ratio:.2f}x, exp ~{expected_ratio:.2f}x)"
            else:
                status = f"⚠️  Slow ({time_ratio:.2f}x, exp ~{expected_ratio:.2f}x)"
        else:
            status = "Baseline"

        height_str = f"{height} ({height_ratio:.2f}x)"
        print(f"{size:<8} {query_time:<12.6f} {len(overlaps):<10} {height_str:<10} {optimal:<10} {status:<35}")

    print("\n✓ Performance benchmark complete")


def stress_test():
    """Stress test with large dataset"""
    print("\n" + "=" * 60)
    print("STRESS TEST")
    print("=" * 60)

    size = 50000
    print(f"\nBuilding tree with {size:,} random intervals...")

    intervals = []
    for _ in range(size):
        x_low = random.randint(0, 100000)
        x_high = random.randint(x_low, x_low + 5000)
        y_low = random.randint(0, 100000)
        y_high = random.randint(y_low, y_low + 5000)
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
    print(f"\nPerforming 100 random queries...")
    total_time = 0
    total_overlaps = 0

    for _ in range(100):
        qx_low = random.randint(0, 90000)
        qx_high = random.randint(qx_low, qx_low + 10000)
        qy_low = random.randint(0, 90000)
        qy_high = random.randint(qy_low, qy_low + 10000)

        start = time.time()
        overlaps = tree.findall_overlapping(tree.root, [qx_low, qx_high, qy_low, qy_high])
        total_time += time.time() - start
        total_overlaps += len(overlaps)

    avg_time = total_time / 100
    avg_overlaps = total_overlaps / 100

    print(f"Average query time: {avg_time:.6f}s")
    print(f"Average overlaps found: {avg_overlaps:.1f}")
    print(f"\n✓ Stress test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("2D AVL INTERVAL TREE - TEST SUITE")
    print("=" * 60)

    # Run all tests
    comprehensive_test_suite()
    performance_benchmark()
    stress_test()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
    print("=" * 60)
    print("\nTo visualize intervals:")
    print("  from visualise_interval_tree import visualize_intervals_2d")
    print("  visualize_intervals_2d([[0,10,0,10], [5,15,5,15]], query=[7,11,7,11])")
    print("\nTo visualize tree structure:")
    print("  from visualise_interval_tree import visualize_tree_structure")
    print("  visualize_tree_structure(tree)")
    print("=" * 60)
