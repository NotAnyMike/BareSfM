import torch
from pdb import set_trace

def projection(img, z, pose, K, K_inv):
    batch_size, _, height, width = img.shape
    x = torch.arange(0, width, 1, requires_grad=False)
    y = torch.arange(0, height, 1, requires_grad=False)

    x, y = torch.meshgrid(x, y)
    x = torch.flatten(x, 0).view(1, -1, 1)
    y = torch.flatten(y, 0).view(1, -1, 1)
    z = torch.flatten(z, 2).view(batch_size, -1, 1)

    # Copy batch wise
    x = x.repeat(batch_size, 1, 1)
    y = y.repeat(batch_size, 1, 1)
    ones = torch.ones(x.shape)

    x = x * z
    y = y * z

    pos = torch.cat([x, y, z, ones], -1).view(batch_size, -1, 4, 1)

    K_inv = K_inv.view(batch_size, 1, 4, 4).expand(-1, int(height * width), -1, -1)
    K = K.view(batch_size, 1, 4, 4).expand(-1, int(height * width), -1, -1)

    # Convert 6DoF to Rotation and translation matrix
    pose = get_extrinsic_matrix(pose)
    pose = pose.view(batch_size, 1, 4, 4).expand(-1, int(height*width), 4, 4)

    world_coords = K_inv @ pos
    proj = K @ pose @ world_coords
    proj = proj / proj[:, :, 2:3, :] # Dividing by z

    # TODO project the pixels from one image to the new one defined by proj
    imgs = []
    for batch in range(batch_size): # TODO can be made better using system map
        x = proj[batch, :, 0]
        y = proj[batch, :, 1]
        imgs.append(bilinear_interpolation(x, y, img[batch]).view(1, height, width, 3))

    imgs = torch.cat(imgs, 0) # adding into a batch
    imgs = imgs.permute(0, 3, 1, 2) # formatting as BCHW

    return imgs

def bilinear_interpolation(x, y, img):
    """
    Computes the bilinear interpolation for x and y.
    x,y should be a flatten non-batch vector. img is a non-batch img
    """
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

    return pix

def get_extrinsic_matrix(pose):
    """
    Returns the rotation matrix representation of the
    rotations and translations from pose.
    """
    return torch.rand((12, 4, 4)) # TODO remove this
