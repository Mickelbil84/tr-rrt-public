import tqdm
import numpy as np
from spatialmath import SE3
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .sdf_mesh import *
from .signed_distance import *
from .conversions import *

PRACTICALLY_INFTY = 1e10

def get_transformed_sampling(m1: SDFMesh, q1: torch.tensor, m2: SDFMesh, q2: torch.tensor, device) -> torch.tensor:
    """
    Transform M2's sampling such that M1 sits (normalized, per the SDF) in the origin. 
    We assume that q1 and q2 are tensors, and if they require grad, then the resulting
    tensor respects the torch computational graph (and can be used to compute grads).

    Returns: Tensor N x 3 of the transformed sampling, where N is number of samples on M2
    """
    m1_centroid = torch.tensor(SE3(*-m1.centroid).A).to(device).float()
    Aq1 = euler_to_matrix(q1, device)
    Aq1 = torch.matmul(m1_centroid, torch.inverse(Aq1))

    Aq2 = euler_to_matrix(q2, device)

    sampling = m2.sampling.T
    row_ones = np.ones((len(m2.sampling),))
    sampling = np.vstack([sampling, row_ones])
    sampling = torch.tensor(sampling).to(device).float()

    transformed_sampling = torch.matmul(torch.matmul(Aq1, Aq2), sampling).T  / m1.max_norm
    return transformed_sampling[:, :3]


def evaluate_sdf_on_sampling(m1: SDFMesh, q1: torch.tensor, m2: SDFMesh, q2: torch.tensor, device) -> torch.tensor:
    """
    Evaluate the (transformed) sampling of M2 on M1's SDF.
    We assume that q1 and q2 are tensors, and if they require grad, then the resulting
    tensor respects the torch computational graph (and can be used to compute grads).

    Returns: Tensor (N,) of distances (in real-world coordinates) for each sample on M2
    """
    transformed_sampling = get_transformed_sampling(m1, q1, m2, q2, device)
    distances = m1.model(transformed_sampling) * -m1.max_norm
    distances = distances.reshape((distances.shape[0],))
    return distances


def get_contact_point_indices(sampling: np.array, distances: np.array, eps_contact: float, eps_cluster: float) -> list:
    """
    Given a sampling array of shape (N, 3), and a corresponding (N,) array
    of (signed) distances, return array of (integer) indices of contact point delegates.
    
    Distances with |d| < eps_contact are considered contact point cadidates.
    We use eps_cluster to cluster together close candidates.
    """
    sampling = np.copy(sampling)
    mask = (distances <= eps_contact)
    indices = np.where(mask)[0]
    return indices

def get_contact_gradients(m1: SDFMesh, q1: torch.tensor, m2: SDFMesh, q2: torch.tensor, eps_contact: float, eps_cluster: float, device, threshold=0.0):
    """
    For each contact point, get the gradient of M1's SDF w.r.t. q2.
    """
    distances = evaluate_sdf_on_sampling(m1, q1, m2, q2, device)
    distances_np = distances.clone().detach().cpu().numpy() - threshold

    A_q2 = xyzrpy_2_SE3(list(q2.clone().detach().cpu().numpy()))

    contact_indices = get_contact_point_indices(m2.sampling, distances_np, eps_contact, eps_cluster)
    
    contact_gradients = []
    for contact_idx in contact_indices:
        m1.model.zero_grad()
        
        val = distances[contact_idx]
        val.backward(retain_graph=True)
        contact_grad = {
            'point': A_q2 * m2.sampling[contact_idx],
            'grad': q2.grad.clone().detach().cpu().numpy(),
            'distance': distances_np[contact_idx] + threshold
        }

        q2.grad.zero_()

        # if contact_grad['distance'] < 0:
        contact_gradients.append(contact_grad)

    return contact_gradients


def get_deepest_gradient(m1: SDFMesh, q1: torch.tensor, m2: SDFMesh, q2: torch.tensor, q2_: torch.tensor, eps_contact: float,
                          eps_cluster: float, device):
    """
    For deepest contact point, get the gradient of M1's SDF w.r.t. q2.
    """
    distances = evaluate_sdf_on_sampling(m1, q1, m2, q2, device)
    distances_np = distances.clone().detach().cpu().numpy()

    distances_ = evaluate_sdf_on_sampling(m1, q1, m2, q2_, device)
    distances_np_ = distances_.clone().detach().cpu().numpy()

    # find indices of sampled points which are very close AFTER the step (hence distances_np_)
    contact_indices = get_contact_point_indices(m2.sampling, distances_np_, eps_contact, eps_cluster)
    if len(contact_indices) == 0:
        return None

    penetrations = distances_np_[contact_indices]-distances_np[contact_indices]
    # if a point is closing in - use its gradient
    # else - use the deepest point
    if np.min(penetrations) < 0:
        gradient_idx = np.argmin(penetrations)
    else:
        gradient_idx = np.argmin(distances_np_[contact_indices])

    m1.model.zero_grad()
    # gradients to be calculated at q2 (rather than q2_)
    val = distances[contact_indices[gradient_idx]]
    val.backward(retain_graph=True)
    gradient = q2.grad.clone().detach().cpu().numpy()
    return gradient

def get_l2_contact_value(m1: SDFMesh, q1: torch.tensor, 
                            m2: SDFMesh, q2: torch.tensor, 
                            eps_contact: float, eps_cluster: float, 
                            device: torch.device, 
                            threshold=0.0):
    """
    Compute the gradient of the L1 loss over the SDF on the contact points
    """
    distances = evaluate_sdf_on_sampling(m1, q1, m2, q2, device)
    distances_np = distances.clone().detach().cpu().numpy() - threshold
    contact_indices = get_contact_point_indices(m2.sampling, distances_np, eps_contact, eps_cluster)

    if len(contact_indices) == 0:
        return None, None

    m1.model.zero_grad()
    val = torch.cat(tuple(distances[idx].reshape((1,)) for idx in contact_indices))
    val = torch.sum(torch.square(val))
    return val, torch.min(distances).detach().cpu().numpy()
    

    

    