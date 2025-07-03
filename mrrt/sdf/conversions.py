import numpy as np


def SE3_2_xyzrpy(T):
    """Convert a 4x4 transformation matrix to xyz+rpy."""
    x, y, z = T[:3, 3]
    sy = -T[2, 0]
    cy = np.sqrt(1 - sy ** 2)
    if cy < 1e-6:
        rx = np.arctan2(-T[1, 2], T[1, 1])
        ry = np.arcsin(sy)
        rz = 0
    else:
        rx = np.arctan2(T[2, 1], T[2, 2])
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(T[1, 0], T[0, 0])
    return [x, y, z, rx, ry, rz]


def xyzrpy_2_SE3(q):
    """Convert xyz+rpy to a 4x4 transformation matrix."""
    x, y, z, rx, ry, rz = q
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def list_2_tensor_with_grad(q, device=None):
    """Compatibility wrapper returning a numpy array."""
    return np.array(q, dtype=float)


def euler_to_matrix(euler_xyz, device=None):
    """Convert xyz+euler angles to a 4x4 transformation matrix."""
    return xyzrpy_2_SE3(euler_xyz)
