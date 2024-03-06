import numpy as np
from scipy.optimize import linprog

from .sdf import *
from .sample import *
from .rotation_utils import *
from .collision_detection import *

DIM = 6

def is_tunnel(
        m1: SDFMesh, q1: np.array, 
        m2: SDFMesh, q2: np.array, 
        delta: float, eps_contact: float, eps_cluster: float,
        device: torch.device):
    """
    Check if M2, when placed in q2, is in a "tunnel", i.e., has very constrained degrees of freedom.
    """
    contacts = get_contact_gradients(
        m1, list_2_tensor_with_grad(q1, device),
        m2, list_2_tensor_with_grad(q2, device), 
        eps_contact, eps_cluster, device)
    k = len(contacts)
    if k < 3:
        print(False, f"k={k}")
        return False
    # print(f"k = {k}")

    c = np.zeros((k,))
    A_ub = []; b_ub = []
    for contact in contacts:
        v = contact['grad']
        a = np.zeros((k,))
        for i in range(k):
            for j in range(DIM - 1):
                a[i] += v[j] * contacts[i]['grad'][j]
        A_ub.append(a)
        b_ub.append(delta - v[DIM - 1])

    A_ub = np.vstack(A_ub)
    b_ub = np.vstack(b_ub)

    res = linprog(c, A_ub, b_ub) 
    print(res.status == 2, f"k={k}")
    return res.status == 2 # infeasible
    

