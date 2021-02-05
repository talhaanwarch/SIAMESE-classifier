# -*- coding: utf-8 -*-
"""
The Resnet mode is taken from Aldaddin Persson github repo
https://github.com/aladdinpersson
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
    
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = SoftPool2D(kernel_size=3, stride=2, padding=1)
        self.num_classes=num_classes
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        if self.num_classes==1:
            x=  self.sigmoid(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


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





def ResNet50(img_channel=1, num_classes=512):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=1, num_classes=512):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=1, num_classes=512):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet101(img_channel=1, num_classes=1000)
    y = net(torch.randn(4, 1, 224, 224)).to("cuda")
    print(y.size())


#test()# -*- coding: utf-8 -*-

