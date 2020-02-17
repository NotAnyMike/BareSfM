import numpy as np
from pdb import set_trace

from dataloaders import Shapes3D_loader, read_lines, parse_lines

def test_read_lines():
    lines = read_lines("test/train_files.txt")
    assert lines == ["1 2", "3 4"]

def test_shapes3d_dataloader():
    dl = Shapes3D_loader('test', True)
    K = np.resize(np.arange(1, 17, dtype=np.float),(4,4))
    assert dl.files == [(1, 2), (3, 4)]
    assert np.array_equal(dl.K, K)

def test_parse_lines():
    correct = [(1., 2.), (2., 3., 3.,)]
    assert parse_liness(["1 2", "2 3 3"],float) == correct
