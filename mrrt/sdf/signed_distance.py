import tqdm
import numpy as np
from spatialmath import SE3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .sdf_mesh import *


def signed_distance_one_directional(m1: SDFMesh, q1: SE3, m2: SDFMesh, q2: SE3, device):
    if m2.sampling is None:
        raise ValueError("SDFMesh (m2) should have sampling")
    
    # move m1 back to original pose
    q = SE3(*-m1.centroid) * q1.inv() * q2
    samples = ((q * m2.sampling.T) / m1.max_norm).T
    samples = torch.from_numpy(samples).float().to(device)
    model_samples = m1.model(samples)
    maxx = torch.max(model_samples)
    val = (-maxx * m1.max_norm).item()
    return val

def signed_distance(m1: SDFMesh, q1: SE3, m2: SDFMesh, q2: SE3, device, single_direction=False):
    """
    Compute signed distance between two meshes in 3D space.
    The transformations q1,q2 \in SE(3) are in the original (un-scaled) model space.
    We take sampling on the second mesh, and compare with the SDF of the first.
    """
    val_1 = signed_distance_one_directional(m1, q1, m2, q2, device)
    if single_direction:
        val_2 = val_1
    else:
        val_2 = signed_distance_one_directional(m2, q2, m1, q1, device)
    return min(val_1, val_2)
