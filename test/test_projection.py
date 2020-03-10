from pdb import set_trace
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from operations import Projection, bilinear_interpolation
from dataloaders import Shapes3D_loader

def test_projection():
    batch_size = 12
    h, w = 280, 320
    dl = Shapes3D_loader(h, w, 'test/test_dataset', True, K_dim=(3,3))
    img = torch.rand((batch_size, 3, h, w))
    depth = torch.rand((batch_size, 1, h, w))
    pose = torch.rand((batch_size, 6))
    K = dl.K.view(1, 3, 3).repeat(batch_size, 1, 1)
    K_inv = dl.K_inv.view(1, 3, 3).repeat(batch_size, 1, 1)
    K = K.view(batch_size, 1, 3, 3).expand(-1, int(h*w), -1, -1)
    K_inv = K_inv.view(batch_size, 1, 3, 3).expand(-1, int(h*w), -1, -1)

    projection = Projection(height=h, width=w, batch_size=batch_size)
    proj = projection(img, depth, pose, K, K_inv)

    assert proj.shape == img.shape

def test_bilinear_interpolation():
    """
    Test 1: image should be the same if the maping is the identity. More
    specifically, it test that at most 0.5% of pixels are at most 0.1 different.
    This is made for pytorch bilinear interpolation function and our own.
    """
    height, width = 280, 320
    dl = Shapes3D_loader(height, width, 'test/test_dataset', True)

    img = dl[0][('color', 'a')]

    x_vec = torch.arange(0, width)
    y_vec = torch.arange(0, height)

    y, x = torch.meshgrid(y_vec, x_vec)
    x_flat = torch.flatten(x).float().view(-1, 1)
    y_flat = torch.flatten(y).float().view(-1, 1)

    img2 = bilinear_interpolation(x_flat, y_flat, img)

    # Comparing with pytorch bilinear interpolation
    x1 = 2*x.float()/width - 1
    y1 = 2*y.float()/height - 1
    grid = torch.cat((x1.view(1, height, width, 1), y1.view(1, height, width, 1)), -1)
    img3 = F.grid_sample(img.view(1, 3, height, width), grid)

    wrong1 = (torch.abs(img - img2) > 0.1).sum().float() / (height*width*3)
    wrong2 = (torch.abs(img - img3) > 0.1).sum().float() / (height*width*3)

    print("\n")
    print("Only %0.2f%% pixels 'significantly' different using OWN method" % (wrong1 * 100))
    print("Only %0.2f%% pixels 'significantly' different using PYTORCH method" % (wrong2 * 100))
    print("\n")

    if wrong1 > 0.05 or wrong2 > 0.05:
        plt.suptitle("Bilinear interpolation sampling methods")
        plt.subplot(131)
        plt.title("org")
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.subplot(132)
        plt.title("own")
        plt.imshow(img2.permute(1, 2, 0))
        plt.subplot(133)
        plt.title("pytorch")
        plt.imshow(img3[0].permute(1, 2, 0).numpy())
        plt.show()

    assert (torch.abs(img-img2) > 0.1).sum() / (height*width*3) < 0.1
