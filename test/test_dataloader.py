import numpy as np
from pdb import set_trace

from dataloaders import Shapes3D_loader, read_lines, parse_lines

def test_read_lines():
    lines = read_lines("test/test_dataset/train_files.txt")
    assert lines == ["1 2", "3 4"]

def test_shapes3d_dataloader():
    dl = Shapes3D_loader(320, 320, 'test/test_dataset', True)
    K = np.resize(np.arange(1, 17, dtype=np.float),(4, 4))
    assert dl.files == [('1', '2'), ('3', '4')]
    assert np.array_equal(dl.K, K)

def test_parse_lines():
    correct = [(1., 2.), (2., 3., 3.,)]
    assert parse_lines(["1 2", "2 3 3"],float) == correct

def test_reshape_and_totensor_transform():
    h, w = 320, 320
    dl = Shapes3D_loader(h, w, 'test/test_dataset', True)
    assert dl[0][('color', 'a')].shape == (3, h, w)

    h, w = 100, 50
    dl = Shapes3D_loader(h, w, 'test/test_dataset', True)
    assert dl[0][('color', 'c')].shape == (3, h, w)
