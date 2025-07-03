import numpy as np
import trimesh
import mcubes


class SDFMesh:
    """Simple SDF representation loaded from an ``*.npz`` file."""

    def __init__(self, mesh_path, device=None, vis=None):
        self.mesh_path = mesh_path
        self.sampling = None
        self.mesh = None
        self.sdf = None
        self.origin = None
        self.voxel_size = None

    def load(self):
        """Load the pre-computed SDF grid from ``mesh_path + '.npz'``."""
        data = np.load(self.mesh_path + '.npz')
        self.sdf = data['sdf']
        self.origin = data['origin']
        self.voxel_size = float(data['voxel_size'])
        self.mesh = trimesh.load(self.mesh_path)
        self.pq = trimesh.proximity.ProximityQuery(self.mesh)

    def generate_sampling(self, num_samples):
        """Generate ``num_samples`` surface points for collision queries."""
        if self.mesh is None:
            self.mesh = trimesh.load(self.mesh_path)
        self.sampling, _ = trimesh.sample.sample_surface(
            self.mesh, num_samples, face_weight=None
        )

    def query(self, points):
        """Tri-linearly interpolate SDF values at given points."""
        pts = np.asarray(points)
        coords = (pts - self.origin) / self.voxel_size
        i0 = np.floor(coords).astype(int)
        d = coords - i0
        max_idx = np.array(self.sdf.shape) - 2
        i0 = np.clip(i0, 0, max_idx)
        i1 = i0 + 1

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        tx, ty, tz = d[:, 0], d[:, 1], d[:, 2]
        g = self.sdf

        c000 = g[x0, y0, z0]
        c100 = g[x1, y0, z0]
        c010 = g[x0, y1, z0]
        c110 = g[x1, y1, z0]
        c001 = g[x0, y0, z1]
        c101 = g[x1, y0, z1]
        c011 = g[x0, y1, z1]
        c111 = g[x1, y1, z1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def fit(self, voxel_size=0.01, padding=0.1):
        """Compute an SDF grid for the mesh and save it as ``mesh_path.npz``."""
        self.mesh = trimesh.load(self.mesh_path)
        bounds = self.mesh.bounds
        b_min = bounds[0] - padding
        b_max = bounds[1] + padding

        xs = np.arange(b_min[0], b_max[0] + voxel_size, voxel_size)
        ys = np.arange(b_min[1], b_max[1] + voxel_size, voxel_size)
        zs = np.arange(b_min[2], b_max[2] + voxel_size, voxel_size)

        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
        points = grid.reshape(-1, 3)

        pq = trimesh.proximity.ProximityQuery(self.mesh)
        sdf_vals = pq.signed_distance(points)
        self.sdf = sdf_vals.reshape(grid.shape[:3])

        self.origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float32)
        self.voxel_size = float(voxel_size)
        np.savez(self.mesh_path + '.npz', sdf=self.sdf, origin=self.origin,
                 voxel_size=self.voxel_size)

    def to_mesh(self, export_path, level=0.0):
        """Export the SDF grid as a mesh using marching cubes."""
        if self.sdf is None:
            raise ValueError('SDF grid was not loaded')
        vertices, triangles = mcubes.marching_cubes(self.sdf, level)
        vertices = vertices * self.voxel_size + self.origin
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(export_path)
        return mesh
