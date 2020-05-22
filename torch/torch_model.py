import torch
import torch.nn as nn

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def conv_layer(ni, nf, ks=3, stride=1,act=True):
    bn = nn.BatchNorm2d(nf)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    act_fn = nn.ReLU(inplace=True)
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

def conv_layer_averpl(ni, nf):
    aver_pl = nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.Sequential(conv_layer(ni, nf), aver_pl)

model = nn.Sequential(
    conv_layer_averpl(1, 64),
    ResBlock(64),
    conv_layer_averpl(64, 64),
    ResBlock(64),
    conv_layer_averpl(64, 128),
    ResBlock(128),
    conv_layer_averpl(128, 256),
    ResBlock(256),
    conv_layer_averpl(256, 512),
    ResBlock(512),
    nn.AdaptiveAvgPool2d((2,2)),
    nn.Flatten(),
    nn.Linear(2048, 40),
    nn.Linear(40, 8)
)