from pdb import set_trace
import matplotlib.pyplot as plt

import torch

from losses import projection, bilinear_interpolation
from dataloaders import Shapes3D_loader

def test_projection():
    batch_size = 12
    h, w = 280, 320
    dl = Shapes3D_loader(h, w, 'test', True)
    img = torch.rand((batch_size, 3, h, w))
    depth = torch.rand((batch_size, 1, h, w))
    pose = torch.rand((batch_size, 6))

    K = dl.K.view(1, 4, 4).repeat(batch_size, 1, 1)
    K_inv = dl.K_inv.view(1, 4, 4).repeat(batch_size, 1, 1)

    projection(img, depth, pose, K, K_inv)
    # TODO
    pass

def test_bilinear_interpolation():
    """ Test 1: image should be the same if the maping is the identity """
    height, width = 280, 320
    dl = Shapes3D_loader(height, width, 'test', True)

    img = dl[0][('color', 'a')]

    x = torch.arange(0, width)
    y = torch.arange(0, height)

    set_trace()
    x, y = torch.meshgrid(x,y)
    x = torch.flatten(x).float().view(-1, 1)
    y = torch.flatten(y).float().view(-1, 1)

    img2 = bilinear_interpolation(x, y, img).view(3, height, width)

    plt.subplot(121)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.subplot(122)
    plt.imshow(img2.permute(1, 2, 0))
    plt.show()

    pass
