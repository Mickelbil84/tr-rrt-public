from .sdf import *
from .distance import *


def union_sample(position_scale=1.0, rotation_scale=1.0):
    q = np.array(SE3_2_xyzrpy(SE3.Rand()))
    q[:3] *= position_scale
    q[3:] *= rotation_scale
    return list(q)


def union_sample_aabb(aabb, rotation_scale=1.0):
    q = np.array(SE3_2_xyzrpy(
        SE3.Rand(
            xrange=(aabb.minx, aabb.maxx),
            yrange=(aabb.miny, aabb.maxy),
            zrange=(aabb.minz, aabb.maxz),
        )))
    q[3:] *= rotation_scale
    return list(q)


class SamplingAABB(object):
    def __init__(self, minx=1e10, maxx=-1e10, miny=1e10, maxy=-1e10, minz=1e10, maxz=-1e10, padding=1.0):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.minz = minz
        self.maxz = maxz
        self.padding = padding

    def update(self, x, y, z):
        self.minx = min(self.minx, x-self.padding)
        self.miny = min(self.miny, y-self.padding)
        self.minz = min(self.minz, z-self.padding)
        self.maxx = max(self.maxx, x+self.padding)
        self.maxy = max(self.maxy, y+self.padding)
        self.maxz = max(self.maxz, z+self.padding)


class Sampler(object):
    def __init__(self):
        self.q = None
        self.aabb = SamplingAABB(minx=-1.0, maxx=1.0, miny=-1.0, maxy=1.0, minz=-1.0, maxz=1.0)

    def sample(self, position_scale, rotation_scale):
        raise NotImplementedError()
    
    def set_bias(self, q):
        raise NotImplementedError()

"""
class TightSampler(Sampler):
    def __init__(self, is_close, minx=-1.0, maxx=1.0, miny=-1.0, maxy=1.0, minz=-1.0, maxz=1.0, n_samples=100):
        self.samples = []
        self.aabb = SamplingAABB(minx=minx, maxx=maxx, miny=miny, maxy=maxy, minz=minz, maxz=maxz, padding=0.0)
        self.node_counts = {}
        self.node_samples = {}
        while len(self.samples) < n_samples:
            q2 = union_sample_aabb(self.aabb, rotation_scale=1.0)
            if is_close(q2):
                self.samples.append(q2)

                for _ in range(1000):
                    s = self.sample()
                    nn = argmin(lambda n: self.distance(n, s), rand_tree)
                    if nn not in node_counts:
                        node_counts[nn] = 0
                        node_samples[nn] = []
                    node_counts[nn] += 1
                    node_samples[nn].append(s)
"""


class UnionSampler(Sampler):
    def sample(self, position_scale, rotation_scale):
        return union_sample(position_scale, rotation_scale)
        # return union_sample_aabb(self.aabb, rotation_scale)

    def set_bias(self, q):
        return

"""
class BiasedTowardsTightSampelr(Sampler):
    def __init__(self, num_tries=5):
        self.q = None
        self.ttl = 0
        self.num_tries = num_tries

    def sample(self, position_scale, rotation_scale):
        if self.q is None:
            return union_sample(position_scale, rotation_scale)
        q = self.q
        print("### I am biased")
        self.ttl -= 1
        if self.ttl == 0:
            print("### Stop bias")
            self.q = None
        return q

    def set_bias(self, q):
        self.q = q
        self.ttl = self.num_tries
"""