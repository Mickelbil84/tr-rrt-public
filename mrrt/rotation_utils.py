import numpy as np

def fix_radians(rad):
    # return ((np.array(rad) + np.pi) % (2*np.pi)) - np.pi
    while rad > np.pi:
        rad -= 2 * np.pi
    while rad < -np.pi:
        rad += 2 * np.pi
    return rad


def fix_configuration(q):
    for i in range(3, 6):
        q[i] = fix_radians(q[i])
    return q

def pave_short_edge(q_start, q_end, n):
    """
    create evenly spread configurations along the sorter side of the rotation
    :return: list of n points (6-configurations)
    """
    q_start = list(q_start)
    q_end = list(q_end)

    for i in range(3,6):
        q_start[i] = fix_radians(q_start[i])
        q_end[i] = fix_radians(q_end[i])

    qs = np.linspace(q_start, q_end, n)
    for i in range(3, 6):
        if q_end[i] - q_start[i] > np.pi:
            qs[:, i] = np.linspace(q_start[i], q_end[i] - 2*np.pi, n)
            qs[:, i] = (qs[:, i] + np.pi) % (2 * np.pi) - np.pi
        elif q_start[i] - q_end[i] > np.pi:
            qs[:, i] = np.linspace(q_start[i], q_end[i] + 2*np.pi, n)
            qs[:, i] = (qs[:, i] + np.pi) % (2 * np.pi) - np.pi
        else:
            pass  # this dimension is spread as needed

    return list(qs)