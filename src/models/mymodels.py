# -*- coding: utf-8 -*-
"""

Softpool layer is taken from 
# https://github.com/qwopqwop200/SoftPool/blob/main/tensorflow_softpool.py#L14-#L22

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
        self.batch=nn.BatchNorm2d(squeeze_channels)
        # expand
        self.expand_1x1 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand_3x3 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3,padding=1)
        

    def forward(self, x):
        x = self.squeeze(x)
        x= self.relu(x)
        x=self.batch(x)
        x = torch.cat([self.expand_1x1(x),self.expand_3x3(x)], dim=1)
        x = self.relu(x)
        return x


class MyModelV1(torch.nn.Module):
    def __init__(self):
        super(MyModelV1, self).__init__()
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
    
    
class MyModelV2(torch.nn.Module):
    
    def __init__(self):
        super(MyModelV2, self).__init__()
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
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 nn.LeakyReLU(),
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
                 nn.LeakyReLU(),
                 nn.Conv2d(in_channels=128*2,out_channels=32,kernel_size=1,stride=1),
                 nn.LeakyReLU(),
                 nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 #nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 nn.Flatten(),
                 #PrintShape(),
                 nn.Linear(784, 512),
                 # nn.Linear(512*2, 512)
                
               
                )        
    def forward(self, x):
        return self.net(x)

class Fire1(nn.Module):
   
    def __init__(self, in_channels, squeeze_channels,expand_channels):
        super(Fire1, self).__init__()

        # squeeze 
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.relu = nn.LeakyReLU()

        # expand
        self.expand_1x1 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand_3x3 =nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3,padding=1)
        

    def forward(self, x):
        x = self.squeeze(x)
        x= self.relu(x)
        x = torch.cat([self.expand_1x1(x),self.expand_3x3(x)], dim=1)
        x = self.relu(x)
        return x
    
class MyModelV4(torch.nn.Module):
    
    def __init__(self):
        super(MyModelV4, self).__init__()
        self.net = torch.nn.Sequential(
                 nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 nn.LeakyReLU(),
                 nn.BatchNorm2d(32),
                 Fire1(in_channels=32, squeeze_channels=16,expand_channels=32),
                 Fire1(in_channels=64, squeeze_channels=16,expand_channels=64),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(128),
                 Fire1(in_channels=128, squeeze_channels=32,expand_channels=96),
                 Fire1(in_channels=192, squeeze_channels=32,expand_channels=128),
                 nn.MaxPool2d(kernel_size=3,stride=2),
                 #nn.BatchNorm2d(256),
                 Fire1(256, 48, 160),
                 Fire1(320, 48, 160),
                 #nn.BatchNorm2d(320),
                 nn.Conv2d(in_channels=320,out_channels=128*2,kernel_size=1,stride=2),
                 nn.LeakyReLU(),
                 nn.Conv2d(in_channels=128*2,out_channels=32,kernel_size=1,stride=1),
                 nn.LeakyReLU(),
                 nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 #nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
                 nn.Flatten(),
                 #PrintShape(),
                 nn.Linear(784, 512),
                 # nn.Linear(512*2, 512)              
               
                )        
    def forward(self, x):
        return self.net(x)
  
    
def mymodel1():
    return MyModelV1()

def mymodel2():
    return MyModelV2()
    
def mymodel3():
    return MyModelV3()

def mymodel4():
    return MyModelV4()


