from pdb import set_trace
import torch

from networks import ResNet_layer, U_net, Decoder

def test_ResNet_layer():
    layer = ResNet_layer(3, 64, 3, 4)
    x = torch.rand((1, 3, 300, 300))
    out = layer(x)
    assert out.shape == (1, 64, 150, 150)

def test_U_net():
    model = U_net(2)
    x = torch.rand((1, 3, 320, 320))
    out = model(x)
    assert len(out) == 2
    assert out[1].shape == (1, 128, int(320/4/2), int(320/4/2))
    assert out[0].shape == (1, 256, int(320/4/2/2), int(320/4/2/2))

def test_decoder():
    encoder = U_net(2)
    decoder = Decoder(2)
    x = torch.rand((1, 3, 320, 320))
    x1 = encoder(x)
    x2 = decoder(x1)

    assert x2.shape == x.shape
