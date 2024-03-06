import numpy as np
import sklearn.neighbors

class MetricNotImplemented(Exception):
    pass

class Metric(object):
    """
    Representation of a metric for nearest neighbor search.
    Should support all kernels/methods for nearest neighbors
    """ 
    def __init__(self):
        pass

    @staticmethod
    def dist(p, q):
        raise MetricNotImplemented('')
   
    @staticmethod
    def sklearn_impl():
        raise MetricNotImplemented('sklearn')


class Metric_Euclidean(Metric):
    """
    Implementation of the Euclidean metric for nearest neighbors search
    """
    @staticmethod
    def dist(p, q):
        return np.linalg.norm(p - q)
    
    @staticmethod
    def sklearn_impl():
        return sklearn.metrics.DistanceMetric.get_metric('euclidean')

class NearestNeighbors(object):
    """
    Abstract class represeting the interface a nearest neighbors algorithm should comply.
    """
    def __init__(self, metric):
        self.metric = metric
        self.points = []

    
    def get_points(self):
        return self.points

    def fit(self, points):
        pass
    
    def k_nearest(self, point, k):
        return []

class NearestNeighborsCached(object):
    """
    Wrapper for a nearest neighbor object that also uses cache,
    and allows for insertion of points in real time.
    This allows adding points lazily, and only when cache is full
    then rebuilding the underlying data structure.

    Example:
        >>> nn = NearestNeighborsCached(NearestNeighbors_sklearn(metric=Metric_Euclidean))
        >>> nn.add_point(Point_d(8, [...]))
    """
    def __init__(self, nn, max_cache=10000):
        self.nn = nn
        self.cache = []
        self.max_cache = max_cache

    def get_points(self):
        return self.nn.points + self.cache

    def fit(self, points):
        if len(points) == 0:
            return
        points = points + self.cache # also push anything in cache
        self.nn.fit(points)
        self.cache.clear()

    def add_point(self, point):
        self.cache.append(point)
        if len(self.cache) == self.max_cache:
            # if cache is full rebuild nn
            points = self.nn.points + self.cache
            self.nn.fit(points)
            self.cache.clear()
    
    def k_nearest(self, point, k):
        res = self.nn.k_nearest(point, k)
        res += self.cache
        res = sorted(res, key=lambda q: self.nn.metric.dist(point, q))
        return res[:k]

class NearestNeighbors_sklearn(NearestNeighbors):
    """
    Sklearn implementation of nearest neighbors
    """
    def __init__(self, metric=Metric_Euclidean):
        super().__init__(metric)
        self.kdtree = None
        self.np_points = []
        self.points = []

    def fit(self, points):
        if len(points) == 0:
            return

        # Convert points to numpy array
        self.points = points
        self.np_points = np.vstack(self.points)
        self.kdtree = sklearn.neighbors.KDTree(self.np_points, metric=self.metric.sklearn_impl())
    
    def k_nearest(self, point, k):
        if self.kdtree is None:
            return []
        
        np_point = point.reshape((1, len(point)))
        _, indices = self.kdtree.query(np_point, k=k)
        res = []
        for idx in indices[0]:
            res.append(self.points[idx])
        return res