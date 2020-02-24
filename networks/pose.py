from pdb import set_trace

import torch
import torch.nn as nn

class Simple_pose(nn.Module):
    """ Simple conv network """
    def __init__(self, height, width, num_layers=4):
        super(Simple_pose, self).__init__()

        assert height % 2 == 0
        assert width % 2 == 0
        assert height % (2**num_layers) == 0
        assert width  % (2**num_layers) == 0

        self.width = width
        self.height = height
        self.num_layers = num_layers

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dilation = 1

        ch_in, ch_out = 6, 64

        self.conv0 = nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3,
                               bias=False, dilation=self.dilation)
        self.relu = lambda : nn.ReLU(inplace=False)
        self.conv1x1 = lambda c_in, c_out: nn.Conv2d(
            c_in, c_out, kernel_size=self.kernel_size, dilation=self.dilation,
            stride=1, padding=self.padding, bias=False)
        self.down_conv = lambda c_in, c_out: nn.Conv2d(
            c_in, c_out, kernel_size=self.kernel_size, dilation=self.dilation,
            stride=self.stride, padding=self.padding, bias=False)

        self.layers = []
        self.layers.append(self.conv0)
        self.layers.append(self.relu())
        for _ in range(num_layers):
            self.layers.append(self.conv1x1(ch_out, ch_out))
            self.layers.append(self.relu())
            ch_in = ch_out
            ch_out = int(ch_out * 2)
            self.layers.append(self.down_conv(ch_in, ch_out))
            self.layers.append(self.relu())

        h = self.height // (2 ** (self.num_layers))
        w = self.width  // (2 ** (self.num_layers))
        self.fc = nn.Linear(ch_out * h * w, 6, bias=True)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        #x = self.model(x)
        for opt in self.layers:
            x = opt(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
