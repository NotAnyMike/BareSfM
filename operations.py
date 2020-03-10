import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms 

from pdb import set_trace

class Projection(nn.Module):
    def __init__(self, *, height, width, batch_size):
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.batch_size = batch_size

        x = torch.arange(0.0, self.width, 1.0, requires_grad=False)
        y = torch.arange(0.0, self.height, 1.0, requires_grad=False)
        x, y = torch.meshgrid(x, y)
        x = x.repeat(batch_size, 1, 1)
        y = y.repeat(batch_size, 1, 1)

        self.x = nn.Parameter(torch.flatten(x, 0).view(self.batch_size, -1, 1))
        self.y = nn.Parameter(torch.flatten(y, 0).view(self.batch_size, -1, 1))

        self.Rt_identity_inv = nn.Parameter(torch.eye(4, 3, requires_grad=False).float())
        self.epsilon = nn.Parameter(torch.tensor([1e-7], requires_grad=False))
        self.dims_parameter = nn.Parameter(torch.tensor([self.width, self.height]).float())
        self.one = nn.Parameter(torch.tensor([1.0]))

    def forward(self, img, z, pose, K, K_inv):
        z = torch.flatten(z, 2).view(self.batch_size, -1, 1)

        x = self.x * z
        y = self.y * z

        pos = torch.cat([x, y, z], -1).view(self.batch_size, -1, 3, 1)

        #K_inv = K_inv.view(self.batch_size, 1, 3, 3).expand(-1, int(self.height * self.width), -1, -1)
        #K = K.view(self.batch_size, 1, 3, 3).expand(-1, int(self.height * self.width), -1, -1)

        # Convert 6DoF to Rotation and translation matrix
        pose = get_extrinsic_matrix(pose)
        pose = pose.view(self.batch_size, 1, 3, 4).expand(-1, int(self.height*self.width), -1, -1)

        world_coords = self.Rt_identity_inv @ K_inv @ pos
        proj = K @ pose @ world_coords
        proj = proj / (proj[:, :, 2:3, :] + self.epsilon) # Dividing by z

        grid = proj.view(self.batch_size, self.height, self.width, 3)[:, :, :, :2]
        grid = 2 * grid / self.dims_parameter - self.one

        imgs = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

        return imgs

def bilinear_interpolation(x, y, img):
    """
    Computes the bilinear interpolation for x and y.
    x,y should be a flatten non-batch vector. img is a non-batch img.
    Keep in mind that x is width here
    """
    Warning("This method should not be use, use instead pytorch grid_sample method")
    _, height, width = img.shape
    x_min = x.int()
    y_min = y.int()

    w_x = x - x_min
    w_y = y - y_min
    
    w1 = torch.cat((w_x, w_y), 1).norm(dim=1)
    w2 = torch.cat((1-w_x, w_y), 1).norm(dim=1)
    w3 = torch.cat((1-w_x, 1-w_y), 1).norm(dim=1)
    w4 = torch.cat((w_x, 1-w_y), 1).norm(dim=1)

    sum_w = w1 + w2 + w3 + w4

    w1 = w1 / sum_w
    w2 = w2 / sum_w
    w3 = w3 / sum_w
    w4 = w4 / sum_w

    # TODO how to deal with out of the sample sampling when the pixel is from outside
    x_min = x_min.clamp(0, width - 2).view(-1)
    y_min = y_min.clamp(0, height - 2).view(-1)

    pix = \
            w1 * img[:, y_min.long(), x_min.long()] + \
            w2 * img[:, y_min.long(), x_min.long() + 1] + \
            w3 * img[:, y_min.long() + 1, x_min.long() + 1] + \
            w4 * img[:, y_min.long() + 1, x_min.long()]

    pix = pix.view(img.shape)
    return pix

def get_extrinsic_matrix(pose):
    """
    Returns the rotation matrix representation of the
    rotations and translations from pose.
    """
    batch_size, _ = pose.shape
    rot = pose[:,:3]
    trans = pose[:,3:]

    rot = transforms.euler_angles_to_matrix(rot,convention="XYZ")
    pose = torch.cat((rot,trans.view(batch_size, 3, 1)), -1)

    return pose
