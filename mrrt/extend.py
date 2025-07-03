import time

import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import silhouette_score

from .sdf import *
from .sample import *
from .rotation_utils import *
from .collision_detection import *


def get_unit_direction(q_start, q_end):
    """
    generate 6-dim vector that reTruepresent a step in the 6-d space in the direction from q_from to q_to,
    while taking care of direction of rotations through the shorter side
    :param q_start:
    :param q_end:
    :return:
    """
    q_from, q_to = np.array(q_start), np.array(q_end)

    # normalize orientations into -pi..pi range
    for i in range(3, 6):
        q_from[i] = fix_radians(q_from[i])
        q_to[i] = fix_radians(q_to[i])

    step = q_to - q_from
    # the above handles xyz correctly, but directions of rotations may need reversal
    for i in range(3, 6):
        if q_to[i] - q_from[i] > np.pi:
            step[i] -= 2*np.pi
        elif q_to[i] - q_from[i] < -np.pi:
            step[i] += 2*np.pi
        else:
            pass  # this dimension is spread as needed

    return step / (np.linalg.norm(step) + 0.0001)


def _proj(v, u):
    return np.dot(v, u) / np.dot(u, u) * u

def _gram_schmidt(contact_points):
    v_ks = [contact['grad'] for contact in contact_points]    
    u_ks = []

    for v_k in v_ks:
        u_k = v_k.copy()
        for u_j in u_ks:
            u_k -= _proj(v_k, u_j)
        u_ks.append(u_k)
    return u_ks


def get_tangent_direction_wiggle(v: np.array,
                          m1: SDFMesh, q1: np.array,
                          m2: SDFMesh, q2: np.array, eta: float,
                          eps_contact: float, eps_cluster: float,
                          device):
    """
    Get a direction v in R6, and return v' which is approximately tangent to the most penetrating two contact points
    """

    num_gradients = 0
    v_ = np.copy(v)
    q2_ = q2 + eta * v_
    grads = []
    for i in range(3):
        deepest_gradient = get_deepest_gradient(
            m1, list_2_tensor_with_grad(q1, device),
            m2, list_2_tensor_with_grad(q2, device),
            list_2_tensor_with_grad(q2_, device),
            eps_contact, eps_cluster, device)
        
        if deepest_gradient is None:
            break

        # Do Grahm-Schmidt on the fly
        for grad in grads:
            deepest_gradient -= _proj(deepest_gradient, grad)
        
        # Never divide by zero
        if np.linalg.norm(deepest_gradient) < 1e-6:
            break

        grads.append(deepest_gradient)

        num_gradients += 1

        v_ -= _proj(v_, deepest_gradient)
        q2_ = q2 + eta * v_

    return v_, num_gradients

def get_tangent_direction_affinity(v: np.array,
                          m1: SDFMesh, q1: np.array,
                          m2: SDFMesh, q2: np.array,
                          eps_contact: float, eps_cluster: float,
                          device):
    """
    Get a direction v in R6, and return v' in TpM
    (where M is the critical manifold for k contact points)
    use average gradient for each cluster, rather than a representative (as in practice there are only 1-3 samples per cluster)
    """
    contact_points = get_contact_gradients(
        m1, list_2_tensor_with_grad(q1, device),
        m2, list_2_tensor_with_grad(q2, device),
        eps_contact, eps_cluster, device)

    if len(contact_points) == 0:
        return v, 0

    # prefer representors of clusters which are deeper in penetration
    # depths = []
    # for cp in contact_points:
    #     depths.append(max(0, -cp['distance']/10))

    # Cluster gradients by similairity
    X = np.vstack([contact['grad'] for contact in contact_points])
    af = AffinityPropagation()
    af.fit(X)
    center_indices = af.cluster_centers_indices_
    if len(center_indices) == 0:
        return v, 0
    all_indices = af.predict(X)

    # get representative contact points for each cluster
    representatives = []
    for idx in center_indices:
        representatives.append(contact_points[idx])

    # averaging over each cluster

    # update gradient for each representative to be gradients of the cluster averaged by penetration depth
    grads = [np.zeros(6) for _ in range(len(representatives))]
    weighted_grads = [np.zeros(6) for _ in range(len(representatives))]
    # sums = [0 for _ in range(len(representatives))]
    # nums = [0 for _ in range(len(representatives))]
    sums = [0] * len(representatives) 
    nums = [0] * len(representatives) 

    for idx in range(len(all_indices)):
        weight = max(0, -contact_points[idx]['distance'])
        sums[all_indices[idx]] += weight
        grads[all_indices[idx]] += contact_points[idx]['grad']
        weighted_grads[all_indices[idx]] += contact_points[idx]['grad'] * weight
        nums[all_indices[idx]] += 1

    # update gradients for cluster representatives with more than one point
    for idx in range(len(representatives)):
        if nums[idx] > 1:
            if sums[idx] > 0.001:
                representatives[idx]['grad'] = weighted_grads[idx] / sums[idx]
            else:
                representatives[idx]['grad'] = grads[idx] / nums[idx]
            if not np.isfinite(representatives[idx]['grad'][0]):
                print('nan')

    # remove (almost) antipodal gradients
    for k in range(len(representatives)-1, 0, -1):
        for kk in range(k):
            if np.dot(representatives[k]['grad'], representatives[kk]['grad']) < -0.9:
                del representatives[k]
                break

    contact_grads = _gram_schmidt(representatives)
    v_ = np.copy(v)
    for grad in contact_grads:
        v_ -= _proj(v_, grad)
    return v_, len(representatives)

def get_tangent_direction_kmeans(v: np.array,
                          m1: SDFMesh, q1: np.array,
                          m2: SDFMesh, q2: np.array,
                          eps_contact: float, eps_cluster: float,
                          device):
    """
    Get a direction v in R6, and return v' in TpM
    (where M is the critical manifold for k contact points)
    use average gradient for each cluster, rather than a representative (as in practice there are only 1-3 samples per cluster)
    """
    contact_points = get_contact_gradients(
        m1, list_2_tensor_with_grad(q1, device),
        m2, list_2_tensor_with_grad(q2, device),
        eps_contact, eps_cluster, device)

    if len(contact_points) == 0:
        return v, 0
    
    X = np.vstack([contact['grad'] for contact in contact_points])

    # Apply k-means for k=1,2,3,4 and find the correct amount
    # Based on https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    best_k = None
    best_sil = None
    best_kmeans = None
    for k in range(2, 4):
        if k > len(contact_points) - 1:
            continue
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
        sil = silhouette_score(X, kmeans.labels_, metric='euclidean')
        if best_sil is None or sil > best_sil:
            best_k = k; best_sil = sil; best_kmeans = kmeans
    
    if best_k is None:
        contact_grads = [contact_points[0]['grad']]
        num_gradients = 1
    else:
        num_gradients = best_k
        contact_grads = best_kmeans.cluster_centers_
        contact_grads = [{'grad': contact_grads[i]} for i in range(contact_grads.shape[0])]
        contact_grads = _gram_schmidt(contact_grads)

    v_ = np.copy(v)
    for grad in contact_grads:
        v_ -= _proj(v_, grad)
    return v_, num_gradients



def retract(m1: SDFMesh, q1: np.array, 
            m2: SDFMesh, q2: np.array, 
            eps_contact: float, eps_cluster: float, lr: float, steps: int,
            device, bv, threshold):
    """
    Retract a vector on TM onto the manifold M
    """
    q1 = list_2_tensor_with_grad(q1, device)
    q2 = list_2_tensor_with_grad(q2, device)

    optimizer = torch.optim.Adam([q2], lr=lr)
    vals = []
    for _ in range(steps):
        # Update bullet viz
        if bv is not None:
            bv.step()

        optimizer.zero_grad()
        m1.model.zero_grad()
        val, dist = get_l2_contact_value(m1, q1, m2, q2, eps_contact, eps_cluster, device)
        if val is None or dist >= threshold/2:
            return q2.detach().cpu().numpy()
        val.backward()
        optimizer.step()
        vals.append(val.detach().cpu().numpy())

    if dist < threshold:
        return None
    return q2.detach().cpu().numpy()


def extend_with_slide(m1, q1, m2, q_near, direction, eta, threshold, eps_contact, device, slide_duration, bv, verbose=True):
    """
    The novel extend function
    """
    eps_cluster = eps_contact * 4

    q1 = np.array(q1)
    q_near = np.array(q_near)

    v = direction

    v = v / np.sqrt(np.dot(v, v))

    qs = []
    num_all_gradients = []
    q2 = q_near
    
    now = time.time()
    for t in range(slide_duration):
        # Update bullet viz
        if bv is not None:
            bv.step()
            bv.set_object_configuration("m2", xyzrpy_2_SE3(q2))

        v_, num_gradients = get_tangent_direction_kmeans(v, m1, q1, m2, q2, eps_contact, eps_cluster, device)
        v_ /= (np.linalg.norm(v_) + 0.00001)

        if t % 20 == 0 and verbose:
            print('# gradients: ', num_gradients)

        q2 = q2 + eta * v_

        last_valid_q = q_near
        if len(qs) > 0:
            last_valid_q = qs[-1]

        q2 = retract(m1, q1, m2, q2, eps_contact, eps_cluster, -threshold, 50, device, bv, threshold)#, last_valid_q)
        if q2 is None:
            if verbose:
                print("give up on current direction")
            v = union_sample()
            v = v / np.sqrt(np.dot(v, v))
            q2 = last_valid_q

        qs.append(q2)
        num_all_gradients.append(num_gradients)

        if len(qs) > 2:
            if distance_between_configurations(qs[-1], qs[-3]) < 0.5 * eta:
                if verbose:
                    print("expand give up - did not advance")
                break

    if verbose:
        print("Took {}[sec]".format(time.time() - now))
    return qs, num_all_gradients
