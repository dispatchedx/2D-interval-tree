"""
2D AVL Interval Tree Implementation (Nested Interval Trees)

A self-balancing interval tree for efficient 2D rectangular region queries.
Uses nested AVL trees: x-axis as primary, y-axis as secondary.
Each x-node contains a y-tree for intervals with that exact x-range.

Space: O(n) - each rectangle stored once
Query: O(log n + k) - augmented tree pruning enables efficient search
"""


class Interval:
    """Represents a 1D interval [low, high]"""

    def __init__(self, low, high):
        self.low = low
        self.high = high


class Node:
    """AVL Node for x-axis interval tree"""

    def __init__(self, interval, y_interval):
        self.interval = interval
        self.max = interval.high
        self.height = 1
        self.left = None
        self.right = None
        self.y_tree = IntervalTree()
        self.y_tree.root = self.y_tree.insert(self.y_tree.root, [y_interval.low, y_interval.high])


class YNode:
    """AVL Node for y-axis interval tree"""

    def __init__(self, interval):
        self.interval = interval
        self.max = interval.high
        self.height = 1
        self.left = None
        self.right = None


class IntervalTree:
    """
    AVL-balanced 2D Interval Tree (Nested Structure)

    Stores 2D rectangular regions and efficiently queries overlaps.
    Each interval is [x_low, x_high, y_low, y_high].

    Implementation follows nested interval tree approach:
    - Primary tree: x-intervals (augmented with max high value)
    - Secondary trees: y-intervals for each unique x-interval

    Example:
        tree = IntervalTree()
        tree.update([[0, 10, 0, 10], [5, 15, 5, 15]])
        overlaps = tree.findall_overlapping(tree.root, [8, 12, 8, 12])
        print(f"Found {len(overlaps)} overlaps")
    """

    def __init__(self, intervals=None):
        """
        Initialize interval tree

        Args:
            intervals: Optional list of [x_low, x_high, y_low, y_high] intervals
        """
        self.root = None
        self.overlaps = []  # Store results for convenience
        if intervals is not None:
            for interval in intervals:
                self.root = self.insert(self.root, interval)

    def get_height(self, node):
        """Get height of node (0 if None)"""
        return 0 if node is None else node.height

    def get_balance(self, node):
        """Get balance factor of node (left height - right height)"""
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def update_height(self, node):
        """Update height of node based on children"""
        if node is not None:
            node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def update_max(self, node):
        """Update max value based on interval and children (for augmented tree)"""
        if node is None:
            return
        node.max = node.interval.high
        if node.left is not None:
            node.max = max(node.max, node.left.max)
        if node.right is not None:
            node.max = max(node.max, node.right.max)

    def rotate_right(self, z):
        """Right rotation for AVL balancing"""
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        self.update_height(z)
        self.update_height(y)
        self.update_max(z)
        self.update_max(y)

        return y

    def rotate_left(self, z):
        """Left rotation for AVL balancing"""
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        self.update_height(z)
        self.update_height(y)
        self.update_max(z)
        self.update_max(y)

        return y

    def insert(self, root, i):
        """
        Insert interval with AVL balancing

        Args:
            root: Current root node
            i: Interval [x_low, x_high, y_low, y_high] or [low, high] for y-tree

        Returns:
            New root after insertion and balancing
        """
        interval = Interval(i[0], i[1])

        # Base case: create new node
        if root is None:
            if len(i) > 2:
                y_interval = Interval(i[2], i[3])
                return Node(interval, y_interval)
            else:
                return YNode(interval)

        # Check for exact x-interval match (add to y-tree only)
        if root.interval.low == interval.low and root.interval.high == interval.high:
            if len(i) > 2:
                y_interval = Interval(i[2], i[3])
                root.y_tree.root = root.y_tree.insert(root.y_tree.root, [y_interval.low, y_interval.high])
            return root

        # BST insert based on low value
        if root.interval.low > interval.low:
            root.left = self.insert(root.left, i)
        else:
            root.right = self.insert(root.right, i)

        # Update height and max (augmented tree)
        self.update_height(root)
        self.update_max(root)

        # Get balance factor and rebalance if needed
        balance = self.get_balance(root)

        # Left Left Case
        if balance > 1 and interval.low < root.left.interval.low:
            return self.rotate_right(root)

        # Right Right Case
        if balance < -1 and interval.low >= root.right.interval.low:
            return self.rotate_left(root)

        # Left Right Case
        if balance > 1 and interval.low >= root.left.interval.low:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)

        # Right Left Case
        if balance < -1 and interval.low < root.right.interval.low:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)

        return root

    def delete(self, root, i):
        """
        Delete interval with AVL rebalancing

        Args:
            root: Current root node
            i: Interval [x_low, x_high, y_low, y_high]

        Returns:
            New root after deletion and balancing
        """
        interval = Interval(i[0], i[1])

        if root is None:
            return root

        # BST delete
        if root.interval.low > interval.low:
            root.left = self.delete(root.left, i)
        elif root.interval.low < interval.low:
            root.right = self.delete(root.right, i)
        elif root.interval.low == interval.low and root.interval.high == interval.high:
            # Try deleting from y-tree first
            if isinstance(root, Node) and len(i) > 2:
                old_y_root = root.y_tree.root
                root.y_tree.root = self.delete(root.y_tree.root, i[2:])
                if root.y_tree.root != old_y_root:
                    # Successfully deleted from y-tree, return without deleting x-node
                    return root

            # Delete x-node if y-tree is empty or it's a y-node
            if root.left is None and root.right is None:
                return None
            elif root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                # Two children: find in-order successor
                successor = root.right
                while successor.left is not None:
                    successor = successor.left
                root.interval = successor.interval
                if isinstance(successor, Node):
                    root.right = self.delete(
                        root.right,
                        [
                            successor.interval.low,
                            successor.interval.high,
                            successor.y_tree.root.interval.low,
                            successor.y_tree.root.interval.high,
                        ],
                    )
                else:
                    root.right = self.delete(
                        root.right,
                        [successor.interval.low, successor.interval.high],
                    )

        if root is None:
            return root

        # Update height and max
        self.update_height(root)
        self.update_max(root)

        # Rebalance
        balance = self.get_balance(root)

        # Left Left
        if balance > 1 and self.get_balance(root.left) >= 0:
            return self.rotate_right(root)

        # Left Right
        if balance > 1 and self.get_balance(root.left) < 0:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)

        # Right Right
        if balance < -1 and self.get_balance(root.right) <= 0:
            return self.rotate_left(root)

        # Right Left
        if balance < -1 and self.get_balance(root.right) > 0:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)

        return root

    def findall_overlapping(self, root, query):
        """
        Find all intervals overlapping with query interval

        Uses augmented tree property (max values) to prune search space.

        Args:
            root: Root of tree
            query: Query interval [x_low, x_high, y_low, y_high]

        Returns:
            List of overlapping intervals as (x_interval, y_interval) tuples
            Also stores result in self.overlaps for convenience
        """
        overlaps = []
        self._findall_overlapping_helper(root, query, overlaps)
        self.overlaps = overlaps
        return overlaps

    def _findall_overlapping_helper(self, root, query, overlaps):
        """Internal recursive helper for overlap search with pruning"""
        if root is None:
            return

        x_interval = Interval(query[0], query[1])

        # Check x-axis overlap: A.low <= B.high AND A.high >= B.low
        if root.interval.low <= x_interval.high and root.interval.high >= x_interval.low:
            # X-intervals overlap, check y-tree
            self._findall_overlapping_y(root.y_tree.root, query, root.interval, overlaps)

        # Pruning using augmented max value:
        # Skip left subtree if query.low > left.max (no intervals can overlap)
        if root.left is not None and x_interval.low <= root.left.max:
            self._findall_overlapping_helper(root.left, query, overlaps)

        # Always search right if query.high >= node.low
        if root.right is not None and x_interval.high >= root.interval.low:
            self._findall_overlapping_helper(root.right, query, overlaps)

    def _findall_overlapping_y(self, root, query, x_interval, overlaps):
        """Search y-tree for overlaps with pruning"""
        if root is None:
            return

        y_interval = Interval(query[2], query[3])

        # Check y-axis overlap
        if root.interval.low <= y_interval.high and root.interval.high >= y_interval.low:
            overlaps.append((x_interval, root.interval))

        # Pruning for y-tree
        if root.left is not None and y_interval.low <= root.left.max:
            self._findall_overlapping_y(root.left, query, x_interval, overlaps)

        if root.right is not None and y_interval.high >= root.interval.low:
            self._findall_overlapping_y(root.right, query, x_interval, overlaps)

    def find_overlapping(self, root, query):
        """
        Find a single interval overlapping with query interval

        Args:
            root: Root of tree
            query: Query interval [x_low, x_high, y_low, y_high]

        Returns:
            First overlapping interval as (x_interval, y_interval) or None
        """
        return self._find_overlapping_helper(root, query)

    def _find_overlapping_helper(self, root, query):
        """Internal recursive helper for finding a single overlap"""
        if root is None:
            return None

        x_interval = Interval(query[0], query[1])

        # Check x-axis overlap
        if root.interval.low <= x_interval.high and root.interval.high >= x_interval.low:
            overlap = self._find_overlapping_y(root.y_tree.root, query[2:4], root.interval)
            if overlap:
                return overlap

        # Search left subtree with pruning
        if root.left is not None and x_interval.low <= root.left.max:
            found = self._find_overlapping_helper(root.left, query)
            if found:
                return found

        return self._find_overlapping_helper(root.right, query)

    def _find_overlapping_y(self, root, y_bounds, x_interval):
        """Search y-tree for a single overlapping interval"""
        if root is None:
            return None

        y_interval = Interval(y_bounds[0], y_bounds[1])

        # Check y-axis overlap
        if root.interval.low <= y_interval.high and root.interval.high >= y_interval.low:
            return (x_interval, root.interval)

        # Search left subtree with pruning
        if root.left is not None and y_interval.low <= root.left.max:
            found = self._find_overlapping_y(root.left, y_bounds, x_interval)
            if found:
                return found

        return self._find_overlapping_y(root.right, y_bounds, x_interval)

    def update(self, intervals):
        """Batch insert multiple intervals"""
        for interval in intervals:
            self.root = self.insert(self.root, interval)


def print_overlaps(overlaps):
    """Print list of overlapping intervals"""
    for x_interval, y_interval in overlaps:
        print(f"  x[{x_interval.low}, {x_interval.high}] × y[{y_interval.low}, {y_interval.high}]")
    print(f"Total: {len(overlaps)} overlaps\n")


if __name__ == "__main__":
    print("=" * 60)
    print("2D AVL Interval Tree - Nested Structure")
    print("=" * 60)

    # Create tree with rectangles
    intervals = [
        [0, 10, 0, 10],  # Large rectangle
        [5, 15, 5, 15],  # Overlapping rectangle
        [20, 30, 20, 30],  # Separate rectangle
        [8, 12, 8, 12],  # Small rectangle
    ]

    tree = IntervalTree(intervals)
    print(f"\nInserted {len(intervals)} rectangles")

    # Query 1: Find all overlaps
    print("\nQuery: Find all rectangles overlapping [7, 11] × [7, 11]")
    overlaps = tree.findall_overlapping(tree.root, [7, 11, 7, 11])
    print_overlaps(overlaps)

    # Query 2: Find single overlap
    print("Query: Find one rectangle overlapping [25, 28] × [25, 28]")
    overlap = tree.find_overlapping(tree.root, [25, 28, 25, 28])
    if overlap:
        x, y = overlap
        print(f"  Found: x[{x.low}, {x.high}] × y[{y.low}, {y.high}]\n")
    else:
        print("  No overlap found\n")

    # Delete an interval
    print("Deleting rectangle [20, 30] × [20, 30]")
    tree.root = tree.delete(tree.root, [20, 30, 20, 30])

    print("Query: Search same region after deletion")
    overlap = tree.find_overlapping(tree.root, [25, 28, 25, 28])
    if overlap:
        x, y = overlap
        print(f"  Found: x[{x.low}, {x.high}] × y[{y.low}, {y.high}]\n")
    else:
        print("  No overlap found\n")

    print("=" * 60)
    print("Run test_interval_tree.py for comprehensive tests")
    print("=" * 60)
