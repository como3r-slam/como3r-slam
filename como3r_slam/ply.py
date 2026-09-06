import numpy as np
from plyfile import PlyData, PlyElement


def save_ply(filename, points, colors):
    """Write an (N, 3) XYZ + (N, 3) RGB point cloud as a binary PLY."""
    colors = colors.astype(np.uint8)
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    PlyData([vertex_element], text=False).write(filename)
