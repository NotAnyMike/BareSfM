from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.opts = OrderedDict()
        
        self.opts['upconv'] = None
        
    def forward(self, x):
        """
        x is a 4d list of outputs from the decoder at different
        resolutions
        """
        return x
