"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn



cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class CompMul1:
    def __init__(self, d1, d2):
        self.fca = nn.Linear(d1, d2)
        self.fcb = nn.Linear(d1, d2)

    def forward(self, x):
        return self.fca(x) * self.fcb(x)


class CompMul2:
    def __init__(self, d1, d2):
        da = (d1+1)//2
        db = d1//2
        assert da + db == d1
        self.fca = nn.Linear(da, d2)
        self.fca = nn.Linear(db, d2)

    def forward(self, x):
        a, b = torch.split(x, 2, dim=-1)
        return self.fca(a) * self.fcb(b)

def linear_act(d1, d2, activation):
    if activation == 'relu':
        return nn.Sequential(
            nn.Linear(d1, d2),
            nn.ReLU(inplace=True)
        )
    elif activation == 'mul1':
        return CompMul2(d1, d2)
    elif activation == 'mul2':
        return CompMul2(d1, d2)


class VGG(nn.Module):

    def __init__(self, features, num_class=100, activation='relu'):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            linear_act(512, 4096, act=activation)
            #nn.Linear(512, 4096),
            #nn.ReLU(inplace=True),
            nn.Dropout(),
            linear_act(4096, 4096, act=activation)
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


