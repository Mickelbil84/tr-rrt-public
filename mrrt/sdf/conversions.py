import torch
from spatialmath import SE3

def SE3_2_xyzrpy(se3):
    return list(se3.t) + list(se3.rpy(order='xyz'))

def xyzrpy_2_SE3(q):
    return SE3(q[:3]) * SE3.RPY(q[3:], order='xyz')

def list_2_tensor_with_grad(q, device):
    return torch.tensor(q).to(device).float().requires_grad_(True)


# try:
#     import pytorch3d.transforms

#     def euler_to_matrix(euler_xyz, device):
#         rotation = pytorch3d.transforms.euler_angles_to_matrix(euler_xyz[3:], 'XYZ')
#         t = pytorch3d.transforms.Transform3d(device=device).rotate(rotation).translate(*euler_xyz[:3])
#         return torch.transpose(t.get_matrix().reshape((4,4)), 0, 1)

# except ImportError:
def euler_to_matrix(euler_xyz, device):
    """
    Convert a tensor of shape (6,) with xyz and euler angles values to a 4x4 corresponding transformation matrix.
    
    Args:
    euler_xyz: torch.Tensor of shape (6,) containing xyz and euler angles values
    
    Returns:
    transform_matrix: torch.Tensor of shape (4, 4) representing the transformation matrix
    """
    
    # Extract xyz and euler angles from the input tensor
    x, y, z, rx, ry, rz = euler_xyz
    
    x = x.reshape((1,))
    y = y.reshape((1,))
    z = z.reshape((1,))
    # Compute sine and cosine of the euler angles
    c_rx, s_rx = torch.cos(rx).reshape((1,)), torch.sin(rx).reshape((1,))
    c_ry, s_ry = torch.cos(ry).reshape((1,)), torch.sin(ry).reshape((1,))
    c_rz, s_rz = torch.cos(rz).reshape((1,)), torch.sin(rz).reshape((1,))
    one = torch.tensor(1.0).reshape((1,)).to(device)
    zero = torch.tensor(0.0).reshape((1,)).to(device)
    
    # Create the rotation matrices
    Rx = torch.cat([
        one, zero, zero, zero,
        zero, c_rx, -s_rx, zero,
        zero, s_rx, c_rx, zero,
        zero, zero, zero, one
    ]).reshape((4,4))
    Ry = torch.cat([
        c_ry, zero, s_ry, zero,
        zero, one, zero, zero,
        -s_ry, zero, c_ry, zero,
        zero, zero, zero, one
    ]).reshape((4,4))
    Rz = torch.cat([
        c_rz, -s_rz, zero, zero,
        s_rz, c_rz, zero, zero, 
        zero, zero, one, zero,
        zero, zero, zero, one
    ]).reshape((4,4))
    R = torch.matmul(torch.matmul(Rx, Ry), Rz).reshape((16,))
    r11, r12, r13, _, r21, r22, r23, _, r31, r32, r33, _, _, _, _, _ = R
    R = torch.cat([
        r33.reshape((1,)), r23.reshape((1,)), r13.reshape((1,)), zero,
        r32.reshape((1,)), r22.reshape((1,)), r12.reshape((1,)), zero,
        r31.reshape((1,)), r21.reshape((1,)), r11.reshape((1,)), zero,
        zero, zero, zero, one
    ]).reshape((4,4))

    # Create the translation vector
    T = torch.cat([
        one, zero, zero, x,
        zero, one, zero, y,
        zero, zero, one, z,
        zero, zero, zero, one
    ]).reshape((4,4))
    
    # Create the transformation matrix
    transform_matrix = torch.matmul(T, R)
    
    return transform_matrix