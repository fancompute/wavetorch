import torch

def src_gaussian(t, t0, tau):
    return torch.exp(-((t-t0)/tau)**2)
    