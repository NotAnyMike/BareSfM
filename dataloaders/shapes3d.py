import numpy as np
import os
from PIL import Image
from pdb import set_trace
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def parse_lines(array, dtype=int):
    """ 
    Converts array from ["1 2", "2 3 3"] into [[1,2],[2,3,3]]
    """
    array = [tuple(dtype(i) for i in line.split()) for line in array]
    return array

def load_img(img_path):
    # Rotations should happen here
    return Image.open(img_path)


class Shapes3D_loader(Dataset):
    def __init__(self,
                 height, width,
                 main_folder,
                 is_train,
                 transform=None,
                 K_dim=(4,4),
                 frames="abc"):
        super(Shapes3D_loader, self).__init__()

        self.height = height
        self.width = width
        self.frames = frames
        self.main_folder = main_folder
        self.is_train = is_train
        self.transform = transform # TODO not used yet
        self.resize = transforms.Resize((self.height, self.width))
        self.normalize = None # TODO not implemented
        self.transforms = transforms.Compose([self.resize, transforms.ToTensor()])
        self.K_dim = K_dim

        files_name = os.path.join(
            self.main_folder,
            'train_files.txt' if self.is_train else 'val_files.txt')

        self.files = parse_lines(read_lines(files_name), str)

        K = os.path.join(main_folder, "intrinsic_matrix.txt")
        K = parse_lines(read_lines(K), float)[0]
        K = np.resize(np.array(K, dtype=np.float),(4, 4))

        if self.K_dim == (3,3):
            K = K[[0,1,3],:3]
            K[2,2] = 1.0
        elif self.K_dim == (4,4):
            pass
        else:
            raise AttributeError("K_dim has an invalid dimension")

        K_inv = np.linalg.pinv(K)
        self.K = torch.from_numpy(K).float()
        self.K_inv = torch.from_numpy(K_inv).float()

    def __len__(self):
        return len(self.files)

    def get_depth(self, idx, frame):
        raise NotImplementedError

    def get_color(self, idx, frame, transform):
        env, img = self.files[idx]
        img_path = os.path.join(self.main_folder, env, "%s_%s.jpg" % (img, frame))
        img = load_img(img_path)
        img = transform(img)
        return img

    def __getitem__(self, idx):
        """
        Returns a dictionary with the follow structure:

        ('color', <frame>): color images
                       'K': Intrinsic matrix
                   'K_inv': Intrinsic matrix inverse
        """
        inputs = {}

        transforms = self.transforms
        for f in self.frames:
            inputs[('color', f)] = self.get_color(idx, f, transforms)

        inputs['K'] = self.K 
        inputs['K_inv'] = self.K_inv

        return inputs
