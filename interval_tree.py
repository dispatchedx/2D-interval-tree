"""
AVL-Balanced 2D Interval Tree Implementation

A self-balancing interval tree for efficient 2D rectangular region queries.
Uses nested AVL trees to maintain O(log n) height on both axes.
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
    AVL-balanced 2D Interval Tree

    Stores 2D rectangular regions and efficiently queries overlaps.
    Each interval is [x_low, x_high, y_low, y_high].

    Example:
        tree = IntervalTree()
        tree.update([[0, 10, 0, 10], [5, 15, 5, 15]])
        tree.findall_overlapping_interval(tree.root, [8, 12, 8, 12])
        print(f"Found {len(tree.overlaps)} overlaps")
    """

    def __init__(self, intervals=None):
        """
        Initialize interval tree

        Args:
            intervals: Optional list of [x_low, x_high, y_low, y_high] intervals
        """
        self.root = None
        self.overlaps = []
        if intervals is not None:
            for interval in intervals:
                self.root = self.insert(self.root, interval)

    def get_height(self, node):
        """Get height of node (0 if None)"""
        if node is None:
            return 0
        return node.height

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
        """Update max value based on interval and children"""
        if node is None:
            return
        node.max = node.interval.high
        if node.left is not None:
            node.max = max(node.max, node.left.max)
        if node.right is not None:
            node.max = max(node.max, node.right.max)

    def rotate_right(self, z):
        r"""
        Right rotation for AVL balancing
             z                       y
            / \                     / \
           y   T4      =>          x   z
          / \                         / \
         x   T3                      T3  T4
        """
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
        r"""
        Left rotation for AVL balancing
           z                           y
          / \                         / \
         T1  y          =>           z   x
            / \                     / \
           T2  x                   T1  T2
        """
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

        # BST insert
        if root.interval.low > interval.low:
            root.left = self.insert(root.left, i)
        else:
            root.right = self.insert(root.right, i)

        # Update height and max
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
                if root.y_tree.root == old_y_root:
                    return root

            # Delete node
            if root.left is None and root.right is None:
                return None
            elif root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                # Two children: find successor
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
                        [
                            successor.interval.low,
                            successor.interval.high,
                            successor.interval.low,
                            successor.interval.high,
                        ],
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

    def findall_overlapping_interval(self, root, i):
        """
        Find all intervals overlapping with query interval

        Args:
            root: Root of tree
            i: Query interval [x_low, x_high, y_low, y_high]

        Returns:
            List of overlapping intervals (stored in self.overlaps)
        """
        self.overlaps = []
        return self._findall_overlapping_interval_helper(root, i)

    def _findall_overlapping_interval_helper(self, root, i):
        """Internal recursive helper for overlap search"""
        interval = Interval(i[0], i[1])
        if root is None:
            return root

        # Check x-axis overlap
        if root.interval.low <= interval.high and root.interval.high >= interval.low:
            self.findall_overlapping_y_interval(root.y_tree.root, i, root.interval)

        if root.left is None and root.right is None:
            return root

        # Search left subtree
        if root.left is None:
            self._findall_overlapping_interval_helper(root.right, i)
        elif interval.low > root.left.max:
            self._findall_overlapping_interval_helper(root.right, i)
        else:
            self._findall_overlapping_interval_helper(root.left, i)
            if interval.high >= root.interval.low:
                self._findall_overlapping_interval_helper(root.right, i)

        return root

    def findall_overlapping_y_interval(self, root, i, x_interval):
        """Search y-tree for overlaps"""
        interval = Interval(i[2], i[3])
        if root is None:
            return root

        # Check y-axis overlap
        if root.interval.low <= interval.high and root.interval.high >= interval.low:
            self.overlaps.append([x_interval, root.interval])

        if root.left is None and root.right is None:
            return root

        # Search y-tree
        if root.left is None:
            self.findall_overlapping_y_interval(root.right, i, x_interval)
        elif interval.low > root.left.max:
            self.findall_overlapping_y_interval(root.right, i, x_interval)
        else:
            self.findall_overlapping_y_interval(root.left, i, x_interval)
            if interval.high >= root.interval.low:
                self.findall_overlapping_y_interval(root.right, i, x_interval)

        return root

    def find_overlapping_interval(self, root, i):
        """
        Find a single interval overlapping with query interval.

        Args:
            root: Root of tree
            i: Query interval [x_low, x_high, y_low, y_high]

        Returns:
            The first overlapping interval found, or None if none exists.
        """
        return self._find_overlapping_interval_helper(root, i)

    def _find_overlapping_interval_helper(self, root, i):
        """Internal recursive helper for finding a single overlap"""
        if root is None:
            return None

        x_interval = Interval(i[0], i[1])
        y_interval_bounds = i[2:4]

        # Check x-axis overlap
        if root.interval.low <= x_interval.high and root.interval.high >= x_interval.low:
            # Check y-tree for overlap
            overlap = self._find_overlapping_y_interval(root.y_tree.root, y_interval_bounds, root.interval)
            if overlap:
                return overlap

        # Decide which subtree to search
        if root.left is not None and x_interval.low <= root.left.max:
            found = self._find_overlapping_interval_helper(root.left, i)
            if found:
                return found

        return self._find_overlapping_interval_helper(root.right, i)

    def _find_overlapping_y_interval(self, root, y_bounds, x_interval):
        """Search y-tree for a single overlapping interval"""
        if root is None:
            return None

        y_interval = Interval(y_bounds[0], y_bounds[1])

        # Check y-axis overlap
        if root.interval.low <= y_interval.high and root.interval.high >= y_interval.low:
            return [x_interval, root.interval]

        # Decide which subtree to search
        if root.left is not None and y_interval.low <= root.left.max:
            found = self._find_overlapping_y_interval(root.left, y_bounds, x_interval)
            if found:
                return found

        return self._find_overlapping_y_interval(root.right, y_bounds, x_interval)

    def update(self, intervals):
        """Batch insert multiple intervals"""
        for interval in intervals:
            self.root = self.insert(self.root, interval)

    def visualize(self):
        """Generate GraphViz visualization of tree structure"""
        from graphviz import Digraph

        def add_x_node(dot, root, parent=None, edge_label=""):
            if root is None:
                return
            node_label = f"X: [{root.interval.low},{root.interval.high}]\\nmax={root.max}\\nh={root.height}"
            dot.node(str(id(root)), node_label)
            if parent is not None:
                dot.edge(str(id(parent)), str(id(root)), label=edge_label)

            add_y_nodes(dot, root.y_tree.root, root, "Y")
            add_x_node(dot, root.left, root, "L")
            add_x_node(dot, root.right, root, "R")

        def add_y_nodes(dot, root, parent_x, edge_label=""):
            if root is None:
                return
            node_label = f"Y: [{root.interval.low},{root.interval.high}]\\nmax={root.max}\\nh={root.height}"
            dot.node(str(id(root)), node_label)
            dot.edge(str(id(parent_x)), str(id(root)), label=edge_label)

            add_y_nodes(dot, root.left, parent_x, "L")
            add_y_nodes(dot, root.right, parent_x, "R")

        dot = Digraph(comment="AVL Interval Tree")
        add_x_node(dot, self.root)
        dot.render("interval_tree", view=True)
        return dot


def print_overlaps(overlaps):
    """Print list of overlapping intervals"""
    for x_interval, y_interval in overlaps:
        print(f"x({x_interval.low} {x_interval.high}) y({y_interval.low} {y_interval.high})")
    print(f"Found: {len(overlaps)} overlaps")


def collect_intervals_with_depth(node, depth=0):
    """
    Recursively collect all 2D intervals from the X-tree and Y-trees
    along with depth (height in X-tree).
    """
    intervals = []
    depths = []

    if node is None:
        return intervals, depths

    # Traverse Y-tree for this X-node
    def collect_y_intervals(y_node):
        if y_node is None:
            return []
        result = [[node.interval.low, node.interval.high, y_node.interval.low, y_node.interval.high]]
        result += collect_y_intervals(y_node.left)
        result += collect_y_intervals(y_node.right)
        return result

    y_intervals = collect_y_intervals(node.y_tree.root)
    intervals += y_intervals
    depths += [depth] * len(y_intervals)

    # Traverse left and right subtrees
    left_intervals, left_depths = collect_intervals_with_depth(node.left, depth + 1)
    right_intervals, right_depths = collect_intervals_with_depth(node.right, depth + 1)

    intervals += left_intervals + right_intervals
    depths += left_depths + right_depths

    return intervals, depths


if __name__ == "__main__":
    print("=" * 60)
    print("2D AVL Interval Tree - Example Usage")
    print("=" * 60)

    # Create tree
    tree = IntervalTree()

    # Insert some rectangles
    print("\nInserting intervals:")
    intervals = [
        [0, 10, 0, 10],  # Large rectangle
        [5, 15, 5, 15],  # Overlapping rectangle
        [20, 30, 20, 30],  # Separate rectangle
        [8, 12, 8, 12],  # Small rectangle inside others
    ]

    for interval in intervals:
        print(f"  [{interval[0]},{interval[1]}] × [{interval[2]},{interval[3]}]")
        tree.root = tree.insert(tree.root, interval)

    # Query 1: Find overlaps with a region
    print("\n" + "-" * 60)
    print("Query 1: Find all rectangles overlapping [7, 11] × [7, 11]")
    tree.findall_overlapping_interval(tree.root, [7, 11, 7, 11])
    print_overlaps(tree.overlaps)

    # Query 2: Non-overlapping region
    print("\n" + "-" * 60)
    print("Query 2: Find all rectangles overlapping [25, 28] × [25, 28]")
    tree.findall_overlapping_interval(tree.root, [25, 28, 25, 28])
    print_overlaps(tree.overlaps)

    # Delete an interval
    print("\n" + "-" * 60)
    print("Deleting interval [20, 30] × [20, 30]")
    tree.root = tree.delete(tree.root, [20, 30, 20, 30])

    # Query again
    print("Query 3: Search same region after deletion")
    tree.findall_overlapping_interval(tree.root, [25, 28, 25, 28])
    print_overlaps(tree.overlaps)
    tree.visualize()

    # Tree statistics
    from test_interval_tree import count_nodes, get_tree_height

    print("\n" + "-" * 60)
    print("Tree Statistics:")
    print(f"  Total nodes: {count_nodes(tree.root)}")
    print(f"  Tree height: {get_tree_height(tree.root)}")
    print(f"  Root balance: {tree.get_balance(tree.root)}")

    print("\n" + "=" * 60)
    print("Example complete! Run test_interval_tree.py for comprehensive tests.")
    print("=" * 60)

    from visualise_interval_tree import visualize_intervals_3d

    intervals, depths = collect_intervals_with_depth(tree.root)
    query = [5000, 5100, 5000, 5100]
    visualize_intervals_3d(intervals, depths, query)
