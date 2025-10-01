"""
Visualization utilities for 2D Interval Tree
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_intervals_2d(intervals, query=None, title="2D Interval Visualization"):
    """
    Visualize 2D intervals as rectangles

    Args:
        intervals: List of [x_low, x_high, y_low, y_high] rectangles
        query: Optional query rectangle to highlight [x_low, x_high, y_low, y_high]
        title: Plot title
    """
    if not intervals:
        print("No intervals to visualize")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw all intervals
    for iv in intervals:
        rect = patches.Rectangle(
            (iv[0], iv[2]),  # bottom-left corner (x_low, y_low)
            iv[1] - iv[0],  # width
            iv[3] - iv[2],  # height
            linewidth=1.5,
            edgecolor="steelblue",
            facecolor="lightblue",
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Add label at center
        cx = (iv[0] + iv[1]) / 2
        cy = (iv[2] + iv[3]) / 2
        ax.text(
            cx, cy, f"[{iv[0]},{iv[1]}]×\n[{iv[2]},{iv[3]}]", ha="center", va="center", fontsize=8, color="darkblue"
        )

    # Draw query rectangle if provided
    if query:
        q_rect = patches.Rectangle(
            (query[0], query[2]),
            query[1] - query[0],
            query[3] - query[2],
            linewidth=2.5,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
        )
        ax.add_patch(q_rect)

        # Add query label
        cx = (query[0] + query[1]) / 2
        cy = (query[2] + query[3]) / 2
        ax.text(cx, cy, "QUERY", ha="center", va="center", fontsize=10, fontweight="bold", color="darkred")

    # Set axis properties
    ax.set_xlabel("X axis", fontsize=12)
    ax.set_ylabel("Y axis", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Calculate bounds with padding
    all_x = [iv[0] for iv in intervals] + [iv[1] for iv in intervals]
    all_y = [iv[2] for iv in intervals] + [iv[3] for iv in intervals]
    if query:
        all_x.extend([query[0], query[1]])
        all_y.extend([query[2], query[3]])

    padding = 5
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def visualize_tree_structure(tree):
    """
    Generate GraphViz visualization of tree structure

    Args:
        tree: IntervalTree instance
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("GraphViz not installed. Install with: pip install graphviz")
        return

    def add_x_node(dot, node, parent=None, edge_label=""):
        """Add x-axis node and its children"""
        if node is None:
            return

        node_id = str(id(node))
        node_label = f"X: [{node.interval.low}, {node.interval.high}]\n" f"max={node.max} | h={node.height}"
        dot.node(node_id, node_label, shape="box", style="filled", fillcolor="lightblue")

        if parent is not None:
            dot.edge(str(id(parent)), node_id, label=edge_label)

        # Add y-tree nodes
        add_y_nodes(dot, node.y_tree.root, node, "Y-tree")

        # Recursively add left and right children
        add_x_node(dot, node.left, node, "L")
        add_x_node(dot, node.right, node, "R")

    def add_y_nodes(dot, node, parent_x, edge_label=""):
        """Add y-axis nodes"""
        if node is None:
            return

        node_id = str(id(node))
        node_label = f"Y: [{node.interval.low}, {node.interval.high}]\n" f"max={node.max} | h={node.height}"
        dot.node(node_id, node_label, shape="ellipse", style="filled", fillcolor="lightgreen")
        dot.edge(str(id(parent_x)), node_id, label=edge_label, style="dashed")

        add_y_nodes(dot, node.left, parent_x, "L")
        add_y_nodes(dot, node.right, parent_x, "R")

    dot = Digraph(comment="2D AVL Interval Tree")
    dot.attr(rankdir="TB")
    add_x_node(dot, tree.root)

    output_file = "interval_tree"
    dot.render(output_file, view=True, cleanup=True)
    print(f"Tree structure saved to {output_file}.pdf")


if __name__ == "__main__":
    # Example usage
    intervals = [
        [0, 10, 0, 10],
        [5, 15, 5, 15],
        [20, 30, 20, 30],
        [8, 12, 8, 12],
    ]

    query = [7, 11, 7, 11]

    visualize_intervals_2d(intervals, query, "Example: 2D Rectangles with Query")
