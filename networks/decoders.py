from collections import OrderedDict
from pdb import set_trace

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
        self.dilation = 1
        self.output_padding = 1

        self.upconv = lambda c_in, c_out: nn.ConvTranspose2d(
            c_in, c_out, kernel_size=self.kernel_size, dilation=self.dilation,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, bias=False)

        ch_in = int(64*(2**self.num_layers))
        ch_out = ch_in // 2

        self.opts = OrderedDict()
        for layer_num in range(num_layers, 0, -1):
            self.opts['upconv%i' % layer_num] = self.upconv(ch_in, ch_out)
            ch_in = ch_out
            ch_out = ch_in // 2

        self.opts['up_ch_conv'] = self.upconv(ch_in, ch_out)
        self.opts['upconv0'] = self.upconv(ch_out, 3)
        
    def forward(self, inputs):
        """
        x is a list of outputs from the decoder at different
        resolutions
        """
        x = inputs[0]
        pointer = self.num_layers
        for key, opt in self.opts.items():
            x = opt(x)
            if pointer > 1:
                pointer -= 1
                x = x + inputs[-pointer]
            x = nn.ReLU(inplace=False)(x)
        return x
