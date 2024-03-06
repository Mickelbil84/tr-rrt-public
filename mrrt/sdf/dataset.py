import os
import time
import random
import pickle

import tqdm
import trimesh
import numpy as np
import pandas as pd
import spatialmath
from spatialmath import SE3

import torch
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    def __init__(self, mesh_path, num_samples, 
        alpha=0.8, boundary_mu=0, boundary_sigma=5e-2, normal_mu=0, normal_sigma=0.7):
        super(SDFDataset, self).__init__()

        # Rescale mesh to unit sphere
        mesh = trimesh.load(mesh_path)
        self.centroid = mesh.centroid
        self.max_norm = np.linalg.norm(np.max(mesh.vertices - mesh.centroid))
        mesh.vertices -= self.centroid
        mesh.vertices /= self.max_norm

        pickle_path = mesh_path + ".pkl"
        if os.path.isfile(pickle_path):
            print("LOADING SAMPLES FROM PICKLE:", pickle_path)
            with open(pickle_path, 'rb') as fp:
                samples, sdf = pickle.load(fp)
        else:

            samples = self._sample_points(mesh, num_samples, 
                alpha, boundary_mu, boundary_sigma, normal_mu, normal_sigma)
            sdf = self._compute_sdf(mesh, samples)

            with open(pickle_path, 'wb') as fp:
                pickle.dump((samples, sdf), fp)
        
        self.num_samples = num_samples
        self.df = pd.DataFrame({
            'x': samples[:, 0],
            'y': samples[:, 1],
            'z': samples[:, 2],
            'sdf': sdf
        })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx % self.num_samples
        
        x = self.df['x'][idx]
        y = self.df['y'][idx]
        z = self.df['z'][idx]
        sdf = self.df['sdf'][idx]

        sample = {
            'xyz': torch.from_numpy(np.array([x,y,z])).float(),
            'sdf': torch.from_numpy(np.array([sdf])).float()
        }
        return sample
    
    def _sample_points(self, mesh, num_samples, alpha, boundary_mu, boundary_sigma, normal_mu, normal_sigma):
        start_time = time.time()
        
        # Sample alpha% points near the mesh
        num_mesh_samples = int(alpha* num_samples)
        samples_near_mesh, _ = trimesh.sample.sample_surface(mesh, num_mesh_samples, face_weight=None)
        for i in range(samples_near_mesh.shape[1]):
            samples_near_mesh[:, i] += np.random.normal(boundary_mu, boundary_sigma, num_mesh_samples)
        
        # Sample (1-alpha)% points in normal Gaussian distribution
        num_normal_samples = num_samples - num_mesh_samples
        samples_normal = np.random.normal(normal_mu, normal_sigma, (num_normal_samples, 3))
        samples = np.concatenate([samples_near_mesh, samples_normal])

        end_time = time.time()
        print("Done sampling: {} [sec]".format(end_time - start_time))

        return samples

    def _compute_sdf(self, mesh, samples):
        start_time = time.time()

        pq = trimesh.proximity.ProximityQuery(mesh)
        batch_size = 5000
        b = 0
        sdf = []
        while b < len(samples):
            print("Batch: {:,}".format(b))
            sdf += list(pq.signed_distance(samples[b:b+batch_size]))
            b += batch_size

        end_time = time.time()
        print("Done computing SDF: {} [sec]".format(end_time - start_time))

        return sdf
    