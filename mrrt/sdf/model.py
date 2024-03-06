import torch
import torch.nn as nn

class SDFModel(nn.Module):
    """
    Generate a Signed Distance Fields neural network which returns
    the SDF for given (x,y,z) position.
    The netowork has N hidden layers of size H each.
    Note that the outputs are in (-1, 1) hence the model should be transformed accordingly.

    Netowrk input -> output:
    (3,) -> (1,)
    (b, 3) -> (b, 1)
    """
    # def __init__(self, N=16, H=64):
    def __init__(self, N=12, H=64):
        super(SDFModel, self).__init__()
        net = [nn.Linear(3, H), nn.ReLU(True)]
        for _ in range(N):
            net.append(nn.Linear(H, H))
            net.append(nn.ReLU(True))
        net.append(nn.Linear(H, 1))
        self.model = nn.Sequential(*net)
    
    def forward(self, xyz):
        return torch.tanh(self.model(xyz))

