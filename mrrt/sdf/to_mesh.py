import tqdm
import mcubes
import trimesh
import numpy as np

import torch

def sdf_to_mesh(model, n, device, original_center, original_scale, export_path):
    eps = 3 / n
    grid = np.zeros((n, n, n))

    # Traverse the volume and compute the SDF
    for i, x in tqdm.tqdm(enumerate(np.arange(-1.5, 1.5, eps)), total=n):
        for j, y in enumerate(np.arange(-1.5, 1.5, eps)):
            # To make things more quick, combine all tested z-values
            # to a single batch to run on the (preferably) GPU
            # (this speeds up computation considerably)
            batch = np.zeros((n, 3))
            for k, z in enumerate(np.arange(-1.5, 1.5, eps)):
                batch[k, 0] = x
                batch[k, 1] = y
                batch[k, 2] = z
            xyz = torch.from_numpy(batch).float().to(device)
            with torch.no_grad():
                val = model(xyz)
            grid[i,j,:] = val.cpu().numpy()[:, 0]
    
    # Compute marching cubes on volume grid
    # (and export if necessary)
    vertices, triangles = mcubes.marching_cubes(grid, 0)
    
    # Export scaled mesh
    mesh = trimesh.Trimesh(vertices, triangles)
    if len(mesh.vertices) == 0:
        print("NO MESH!!")
        return
    
    mesh.vertices -= mesh.centroid
    mesh.vertices /= np.linalg.norm(np.max(mesh.vertices))
    mesh.vertices *= original_scale
    mesh.vertices += original_center

    mesh.export(export_path)

    return vertices, triangles
    


    