"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:32
@File : model.py
"""
from __init__ import *


class CNN20DModel(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(3, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)))
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, (5, 3), padding=(3, 1)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)),
                                    nn.Flatten(),
                                    nn.Dropout(p=0.5))
        self.fc = nn.Linear(46080, out_features)
        self.model = nn.Sequential(self.block1,
                                   self.block2,
                                   self.block3,
                                   self.fc)

    def forward(self, x):
        return self.model(x)


# class Res20DMODEL(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         ...
#
#     def forward(self):
#         ...
