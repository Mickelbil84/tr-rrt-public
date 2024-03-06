import platform

import torch
import visdom

import mrrt
import mrrt.sdf

MACOS_USE_MPS = False
if platform.system() == "Darwin" and MACOS_USE_MPS:
    # Test for M1 GPUs
    device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

puzzle_name = '09301'

if __name__ == "__main__":
    vis = visdom.Visdom(env='sdf')
    mesh = mrrt.sdf.SDFMesh("./resources/models/joint_assembly_rotation/general/{}/0.obj".format(puzzle_name), device, vis)
    mesh.fit(num_samples=1000000, num_epochs=100)
    mesh.load()
    mesh.to_mesh("./resources/models/joint_assembly_rotation/general/{}/0_sdf.obj".format(puzzle_name), n=500)

    mesh = mrrt.sdf.SDFMesh("./resources/models/joint_assembly_rotation/general/{}/1.obj".format(puzzle_name), device, vis)
    mesh.fit(num_samples=1000000, num_epochs=2)
    mesh.load()
    mesh.to_mesh("./resources/models/joint_assembly_rotation/general/{}/1_sdf.obj".format(puzzle_name), n=500)
