import numpy as np
import os
from PIL import Image
from pdb import set_trace
from torch.utils.data import Dataset

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
    return Image.load(img_path)


class Shapes3D_loader(Dataset):
    def __init__(self,
                 main_folder,
                 is_train,
                 transform=None,
                 frames="abc"):
        super(Shapes3D_loader, self).__init__()

        self.main_folder = main_folder
        self.is_train = is_train
        self.transform = transform

        files_name = os.path.join(
            self.main_folder,
            'train_files.txt' if self.is_train else 'val_files.txt')

        self.files = parse_lines(read_lines(files_name), str)

        K = os.path.join(main_folder, "intrinsic.txt")
        K = parse_lines(read_lines(K), float)[0]
        K = np.resize(np.array(K, dtype=np.float),(4,4))
        self.K = K

    def __len__(self):
        return len(self.files)

    def get_depth(self, idx, frame):
        raise NotImplementedError

    def get_color(self, idx, frame):
        env, img = self.files[idx]
        img_path = os.path.join(self.main_folder, env, "%s_%s.jpg" % (img, frame))
        return load_img(img_path)

    def __get_item__(self,idx):
        """
        Returns a dictionary with the follow structure:

        ('color', <frame>): color images
                       'K': Intrinsic matrix
                   'K_inv': Intrinsic matrix inverse
        """
        inputs = {}

        for f in self.frames:
            inputs[('color', f)] = self.get_color(idx, frame)

        K_inv = np.linalg.pinv(K)
        inputs['K'] = torch.from_numpy(K).float()
        inputs['K'] = torch.from_numpy(K_inv).float()

        return inputs
