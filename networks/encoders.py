from pdb import set_trace
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # 18, 34, 50, 101, 152
        self.resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
        raise NotImplementedError("This class has not been implemented")

    def forward(self, x):
        """ Returns the output during different compressions """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)

        # Commented to keep square result
        # TODO check if correct
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x) # Fully connected layer
        return x, x3, x2, x1


class ResNet_layer(nn.Module):
    """
    Basic Layer of a resnet

    ch_in: number of channels in
    kernel_size: int of dimention of kernel
    depth: how many conv will it have
    """
    def __init__(self, ch_in, ch_out, kernel_size, depth):
        super(ResNet_layer, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.opts = OrderedDict()

        self.downconv = lambda c_in, c_out: nn.Conv2d(
            c_in, c_out,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False)
        self.conv = lambda c_in, c_out: nn.Conv2d(
            c_in, c_out,
            kernel_size=self.kernel_size,
            stride=1, padding=1, dilation=1, bias=False)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, padding=0)
        self.relu = lambda: nn.ReLU(inplace=False)

        for d in range(1, depth + 1):
            if d == 1:
                # downscale
                self.opts['downconv%i' % d] = self.downconv(ch_in, self.ch_out)
            else:
                # keep dims same
                self.opts['conv%i' % d] = self.conv(ch_in, self.ch_out)

            # The last one has to have a connection before reluing
            if d+1 != depth:
                self.opts['relu%i' % d] = self.relu()

            ch_in = self.ch_out

        self.model = nn.Sequential(self.opts)

    def forward(self, x):
        identity = self.conv1x1(x)

        x = self.model(x)

        x = x + identity
        x = self.relu()(x)

        return x


class U_net(nn.Module):
    """
    Based on ResNet 18, but not using the pretrained network on pytorch hub.
    Can be considered a simplified version of resnet18.
    We are not using any normalisation layer
    """
    def __init__(self, num_layers=4):
        super(U_net, self).__init__()
        self.num_layers = num_layers

        # few layers following ResNet 18
        self.layers = OrderedDict()

        self.maxpool = lambda: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ch_in, ch_out = 3, 64

        self.layers['conv0'] = nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=2, padding=3)
        self.layers['relu0'] = nn.ReLU(inplace=False)
        self.layers['maxpool0'] = self.maxpool()
        for layer_num in range(self.num_layers):
            ch_in = ch_out
            ch_out *= 2

            layer = ResNet_layer(ch_in, ch_out, 3, 2)
            self.layers['layer%i' % (layer_num + 1)] = layer

        # TODO Consider deleting these two extra operations
        #self.layers['avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
        #self.layers['fc'] = nn.Linear(ch_out, self.output_size)

    def forward(self, x):
        """
        Returns self.num_layers outputs 
        From last to first. THe first element returned is the last layer.
        """
        outputs = []
        for key, opt in self.layers.items():
            x = opt(x)
            if 'layer' in key:
                outputs.append(x)
        outputs[-1] = x # the last layer has more opt to pass through
        return outputs[::-1]
