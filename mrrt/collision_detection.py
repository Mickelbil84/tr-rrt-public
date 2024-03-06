from .sdf import *

from .distance import *
from .rotation_utils import *

def is_collision(mesh1, q1, mesh2, q2, threshold, device):
    dist = distance_between_objects(mesh1, q1, mesh2, q2, device)
    return dist < threshold

def is_edge_valid(mesh_static, q_static, mesh_dynamic, q_start, q_end, n, threshold, device):
    """
    varify validity (clearance) of n points evenly spread along the edge from q_start to q_end
    :param q_start: starting point (configuration)
    :param q_end: end point
    :param n: number of check points along the edge
    :return: bool
    """
    qs = pave_short_edge(q_start, q_end, n)

    for q in qs:
        if is_collision(mesh_static, q_static, mesh_dynamic, q, threshold, device):
            return False

    return True