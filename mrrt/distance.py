import numpy as np

from .sdf import signed_distance

def distance_between_objects(mesh1, q1, mesh2, q2, device=None, single_direction=False):
    """
    actual distance between two pieces (mesh1, mesh2) when located in configurations q1 and q2 respectively
    :param mesh1:
    :param q1:
    :param mesh2:
    :param q2:
    :return: float. distance between units
    """
    return signed_distance(mesh1, q1, mesh2, q2, single_direction)

def distance_between_configurations(q1: list, q2: list):
    """
    how far away are two configurations (of combination of the two pieces)
    :param q1:
    :param q2:
    :return:
    """
    values_1 = np.array(q1)
    values_2 = np.array(q2)
    dists = np.abs(values_1 - values_2)
    # for the rotations, find the shortest angle for each axis
    for i in range(3, 6):
        dists[i] = dists[i] % (2*np.pi)
        if dists[i] > np.pi:
            dists[i] = 2 * np.pi - dists[i]
        # weigh rotations lower to generate more rotations - randomization range of rotation is 2*pi bigger
        dists[i] /= (np.pi)
    return np.linalg.norm(dists)
