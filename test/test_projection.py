import torch

from losses import projection

def test_projection():
    h, w = 320, 280
    img = torch.rand((12, 3, h, w))
    depth = torch.rand((12, 1, h, 2))
    pose = torch.rand((12, 6))
    projection(img, depth, pose)
    pass
