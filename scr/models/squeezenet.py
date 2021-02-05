# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:44:21 2021

@author: TAC
"""


import torch
import torch.nn as nn

class Fire(nn.Module):
   
    def __init__(self, in_channels, squeeze_channels,expand_channels):
        super(Fire, self).__init__()

        # squeeze 
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU()

        # expand
        self.expand_1x1 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand_3x3 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3,padding=1)
        

    def forward(self, x):
        x = self.squeeze(x)
        x= self.relu(x)
        x = torch.cat([self.expand_1x1(x),self.expand_3x3(x)], dim=1)
        x = self.relu(x)
        return x

class SqueezeNetV1(torch.nn.Module):
    """THe originial paper author"""
    def __init__(self):
        super(SqueezeNetV1, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64),
                Fire(128, 16, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128),
                Fire(256, 32, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192),
                Fire(384, 48, 192),
                Fire(384, 64, 256),
                Fire(512, 64, 256),
                
                nn.Dropout(p=0.5),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                #PrintShape(),
        
               
                )        
    def forward(self, x):
        return self.net(x)    
    
    