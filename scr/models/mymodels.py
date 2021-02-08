# -*- coding: utf-8 -*-
"""

Softpool layer is taken from 
# https://github.com/qwopqwop200/SoftPool/blob/main/tensorflow_softpool.py#L14-#L22
Squeezenet model is taken from orignal paper repo
https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1
"""

import torch
import torch.nn as nn


class SoftPool2D(nn.Module):
    def __init__(self,kernel_size=3, stride=2, padding=1):
        super(SoftPool2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
    


class PrintShape(nn.Module):
    """This is for debugging purpose"""
    def __init__(self):
        super(PrintShape, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
    

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



   
    


class SqueezeNetV2(torch.nn.Module):
    def __init__(self):
        super(SqueezeNetV2, self).__init__()
        self.net = torch.nn.Sequential(
                 nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 Fire(in_channels=32, squeeze_channels=16,expand_channels=32),
                 Fire(in_channels=64, squeeze_channels=16,expand_channels=64),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(128),
                 Fire(in_channels=128, squeeze_channels=32,expand_channels=96),
                 Fire(in_channels=192, squeeze_channels=32,expand_channels=128),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(256),
                 Fire(256, 48, 160),
                 Fire(320, 48, 160),
                 #nn.BatchNorm2d(320),
                 nn.Conv2d(in_channels=320,out_channels=384,kernel_size=3,stride=2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=384,out_channels=512,kernel_size=3,stride=2),
                 nn.Flatten(),
                 nn.Linear(512*4, 512*2),
                 nn.Linear(512*2, 512)
                
               
                )        
    def forward(self, x):
        return self.net(x)    
    
    
class SqueezeNetV3(torch.nn.Module):
    
    def __init__(self):
        super(SqueezeNetV3, self).__init__()
        self.net = torch.nn.Sequential(
                 nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 Fire(in_channels=32, squeeze_channels=16,expand_channels=32),
                 Fire(in_channels=64, squeeze_channels=16,expand_channels=64),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(128),
                 Fire(in_channels=128, squeeze_channels=32,expand_channels=96),
                 Fire(in_channels=192, squeeze_channels=32,expand_channels=128),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(256),
                 Fire(256, 48, 160),
                 Fire(320, 48, 160),
                 #nn.BatchNorm2d(320),
                 nn.Conv2d(in_channels=320,out_channels=128*2,kernel_size=1,stride=2),
                 nn.Conv2d(in_channels=128*2,out_channels=32,kernel_size=1,stride=1),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 #nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 nn.Flatten(),
                 #PrintShape(),
                 nn.Linear(784, 512),
                 # nn.Linear(512*2, 512)
                
               
                )        
    def forward(self, x):
        return self.net(x)


class MyModelV3(torch.nn.Module):
    
    def __init__(self):
        super(MyModelV3, self).__init__()
        self.net = torch.nn.Sequential(
                 nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2),
                 SoftPool2D(kernel_size=3,stride=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 Fire(in_channels=32, squeeze_channels=16,expand_channels=32),
                 Fire(in_channels=64, squeeze_channels=16,expand_channels=64),
                 SoftPool2D(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(128),
                 Fire(in_channels=128, squeeze_channels=32,expand_channels=96),
                 Fire(in_channels=192, squeeze_channels=32,expand_channels=128),
                 SoftPool2D(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(256),
                 Fire(256, 48, 160),
                 Fire(320, 48, 160),
                 #nn.BatchNorm2d(320),
                 nn.Conv2d(in_channels=320,out_channels=128*2,kernel_size=1,stride=2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=128*2,out_channels=32,kernel_size=1,stride=1),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 #nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 nn.Flatten(),
                 #PrintShape(),
                 nn.Linear(784, 512),
                 # nn.Linear(512*2, 512)
                
               
                )        
    def forward(self, x):
        return self.net(x)

def mymodel():
    return MyModelV3()

def mymodel1():
    return SqueezeNetV3()

