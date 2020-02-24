import torch
from pdb import set_trace

def projection(img, depth, pose):
    _, _, height, width = img.shape
    x = torch.arange(0, width, 1, requires_grad=False)
    y = torch.arange(0, height, 1, requires_grad=False)

    x, y = torch.meshgrid(x, y)
    x = torch.flatten(x, 0)
    y = torch.flatten(y, 0)
    depth = torch.flatten(depth, 2)
