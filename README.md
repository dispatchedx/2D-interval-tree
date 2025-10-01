# 2D AVL Interval Tree

This project is a from-scratch Python implementation of a self-balancing 2D Interval Tree using Augmented AVL Tree. It is designed to efficiently store and query 2D rectangular regions (e.g., `[x_low, x_high, y_low, y_high]`) for overlaps.

The data structure uses nested AVL trees to maintain balance, ensuring that insertion, deletion, and query operations have a time complexity of **O(log n + k)**, where `n` is the number of intervals stored and `k` is the number of reported overlaps.

This 2D implementation builds on that by:
1. Creating a primary Interval Tree based on the **x-intervals** of the rectangles.
2. For each unique x-interval in the tree, creating a secondary Interval Tree to store all **y-intervals** that share that same x-interval.

## Algorithmic Complexity
- **Space Complexity:** O(n) - each rectangle stored once
- **Build Time:** O(n log n) - n insertions, each O(log n)
- **Query Time (Single Overlap):** O(log n) - search both x and y trees
- **Query Time (All Overlaps):** O(log n + k) - where k is the number of overlaps found
- **Insertion/Deletion:** O(log n) - AVL-balanced trees maintain logarithmic height

## Features

- **AVL Balancing:** The tree is self-balancing on the primary (x-axis) and all secondary (y-axis) trees, preventing worst-case performance scenarios with sorted or skewed data.
- **Efficient 2D Overlap Queries:**
  - `findall_overlapping`: Finds all rectangular intervals that overlap with a given query rectangle.
  - `find_overlapping`: Finds and returns the first overlapping interval, useful when only one match is needed.
- **Core Operations:** Supports insertion and deletion of intervals while maintaining the tree's balance and integrity.
- **Comprehensive Test Suite:** Includes a full suite of tests covering correctness, AVL properties, performance benchmarks, and a high-load stress test.
- **Visualization Tools:** Comes with utilities to visually inspect the data and the tree structure:
  - `visualize_intervals_2d`: Plots the stored rectangles and a query rectangle using `matplotlib`.
  - `visualize_tree_structure`: Generates a graph of the nested tree structure using `graphviz`.

## Requirements

The core implementation has **no external dependencies**. For visualization and testing, you will need:

- `matplotlib`
- `graphviz` (both the Python library and the Graphviz system package)

You can install the Python libraries using `pip`:

```bash
pip install matplotlib graphviz
```

## How to Use

Below is a basic example of how to create a tree, insert intervals, and perform an overlap query.

```python
from interval_tree import IntervalTree, print_overlaps

# 1. Define the rectangular intervals [x_low, x_high, y_low, y_high]
intervals = [
    [0, 10, 0, 10],    # Large rectangle
    [5, 15, 5, 15],    # Overlapping rectangle
    [20, 30, 20, 30],  # Separate rectangle
    [8, 12, 8, 12],    # Small rectangle
]

# 2. Create an IntervalTree instance
tree = IntervalTree(intervals)
print(f"Inserted {len(intervals)} rectangles.")

# 3. Define a query rectangle and find all overlaps
query_rect = [7, 11, 7, 11]
print(f"\nQuery: Find all rectangles overlapping with {query_rect}")

overlaps = tree.findall_overlapping(tree.root, query_rect)

# 4. Print the results
print_overlaps(overlaps)

# Example of deleting an interval
print("Deleting rectangle [20, 30] × [20, 30]")
tree.root = tree.delete(tree.root, [20, 30, 20, 30])
```

## Running Tests

A comprehensive test suite is provided in `test_interval_tree.py`. To run all correctness checks, performance benchmarks, and the stress test, simply execute the file:

```bash
python test_interval_tree.py
```
