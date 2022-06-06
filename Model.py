
from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet2(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(ConvNet2, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

    def forward(self, x):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
