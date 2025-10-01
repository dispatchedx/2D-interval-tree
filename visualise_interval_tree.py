import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_intervals_3d(intervals, depths=None, query=None):
    """
    Visualize 2D intervals as 3D boxes.

    Args:
        intervals: List of [x_low, x_high, y_low, y_high]
        depths: Optional list of z values (e.g., tree depth per interval)
        query: Optional query rectangle to highlight [x_low, x_high, y_low, y_high]
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if depths is None:
        depths = [0] * len(intervals)

    for iv, z in zip(intervals, depths):
        x0, x1, y0, y1 = iv
        # Define vertices of the rectangle prism at height z
        verts = [
            [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],  # bottom
            [(x0, y0, z + 0.5), (x1, y0, z + 0.5), (x1, y1, z + 0.5), (x0, y1, z + 0.5)],  # top
        ]
        for vert in verts:
            poly = Poly3DCollection([vert], facecolors="blue", edgecolors="black", alpha=0.3)
            ax.add_collection3d(poly)

    if query:
        x0, x1, y0, y1 = query
        z = max(depths) + 0.5 if depths else 0
        verts = [
            [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
            [(x0, y0, z + 0.5), (x1, y0, z + 0.5), (x1, y1, z + 0.5), (x0, y1, z + 0.5)],
        ]
        for vert in verts:
            poly = Poly3DCollection([vert], facecolors="red", edgecolors="black", alpha=0.6)
            ax.add_collection3d(poly)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Depth / Tree Level")
    ax.set_title("3D Interval Visualization")
    ax.set_xlim(0, max(iv[1] for iv in intervals) + 10)
    ax.set_ylim(0, max(iv[3] for iv in intervals) + 10)
    ax.set_zlim(0, max(depths) + 2)
    plt.show()
