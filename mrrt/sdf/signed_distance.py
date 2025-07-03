import numpy as np

from .sdf_mesh import SDFMesh
from .conversions import xyzrpy_2_SE3


def transform_points(T, points):
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]


def signed_distance_one_directional(m1: SDFMesh, T1, m2: SDFMesh, T2):
    if m2.sampling is None:
        raise ValueError("SDFMesh (m2) should have sampling")
    T = np.linalg.inv(T1) @ T2
    samples = transform_points(T, m2.sampling)
    distances = m1.query(samples)
    return -np.max(distances)


def signed_distance(m1: SDFMesh, q1, m2: SDFMesh, q2, single_direction=False):
    """Compute signed distance between two meshes using pre-computed SDF grids."""
    T1 = xyzrpy_2_SE3(q1)
    T2 = xyzrpy_2_SE3(q2)
    val_1 = signed_distance_one_directional(m1, T1, m2, T2)
    if single_direction:
        val_2 = val_1
    else:
        val_2 = signed_distance_one_directional(m2, T2, m1, T1)
    return min(val_1, val_2)
