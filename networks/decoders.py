from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ num_layers: the number of layers """
    def __init__(self, num_layers=4):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.stride = 2
        self.kernel_size = 3
        self.padding = 1

        self.upconv = lambda c_in, c_out: nn.ConvTranspose2d(c_in, c_out,
                kernel_size=self.kernel_size, stride=self.stride,
                padding=self.padding, bias=False)

        ch_in = 64*(2**(self.num_layers + 1))
        ch_out = int(ch_in / 2)

        self.opts = OrderedDict()
        for layer_num in range(num_layers, 0, -1):
            self.opts['upconv%i' % layer_num] = self.upconv(ch_in, ch_out)
            ch_in = ch_out
            ch_out /= 2

        self.opts['up_ch_conv'] = nn.ConvTranspose2d(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        ch_in /= 2
        ch_out /= 2
        self.opts['upconv0'] = self.upconv(ch_in, ch_out)
        
    def forward(self, inputs):
        """
        x is a list of outputs from the decoder at different
        resolutions
        """
        x = inputs[0]
        pointer = self.num_layers
        for opt in self.opts:
            pass
        return x
