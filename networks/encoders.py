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


class ResNet_layer(nn.Model):
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
        self.dilatation = 1
        self.opts = OrderedDict()

        self.conv_fn = lambda c_in, c_out: nn.Conv2d(
            c_in, c_out,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False)
        self.relu_fn = lambda: nn.ReLU(inplace=False)

        for d in range(depth):
            self.opts['conv%i' % d] = self.conv_fn(ch_in, self.ch_out)
            if d+1 != depth:
                self.opts['relu%i' % d] = self.relu_fn()
            ch_in = self.ch_out

        self.model = nn.Sequential(self.opts)

    def forward(self, x):
        identity = x

        x = self.model(x)

        x = x + identity
        x = self.relu(x)

        return x

def conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, dilatation=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, dilatation, bias=False)


class U_net(nn.Model):
    """ Based on ResNet 18, but not using the pretrained network on pytorch hub """
    def __init__(self):
        # few layers following ResNet 18
        self.layers = []

        self.maxpool = nn.MaxPool2d(3, 2)

        layer = nn.Conv2d(3, 64, 7, 2)
        self.layers.append(layer)
        for layer_num in range(1,5):
            layer = nn.Conv2d()
            self.layers.append(layer)

    def forward(self, x):
        """ returns four outputs """
        x = self.maxpool(x)
        raise NotImplementedError
