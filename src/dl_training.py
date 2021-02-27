# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:01:34 2021
kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
@author: TAC
"""
#import packages
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from glob import glob    

#gpu or cpu
device='cuda' if torch.cuda.is_available() else 'cpu'


#load data



#calculate mean and std of data
# dataset.imgs
# mean,std=[],[]
# for path in dataset.imgs:
#     img=Image.open(path[0]).convert("L")
#     mean.append(np.mean(img))
#     std.append(np.std(img))
# print(np.mean(mean)/255)    #0.20
# print(np.mean(std)/255)    #0.21

from albumentations import *

from albumentations.pytorch import ToTensorV2

def augmentation1():
    return Compose([
            RandomResizedCrop(248, 248),
            #Transpose(p=0.5),
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            ShiftScaleRotate(p=0.5),
            GaussNoise() ,
            #IAASuperpixels(),
            Normalize(mean=[0.5,], std=[0.5,], p=1.0),
            #CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  



#image augmentation
from torchvision import transforms
def augmentation():
    aug=transforms.Compose([
        transforms.Resize(size=(248,248)),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
        transforms.RandomRotation(degrees=30),
        #transforms.RandomPerspective(distortion_scale=0.1), 
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.RandomErasing(p=0.5), #working
        transforms.Normalize((0.5, ), (0.5, )),
                          ])
    return aug

def no_augmentation():
    aug=transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
                          ])
    return aug

from torch.utils.data.dataloader import DataLoader
from dataloader import SiameseNetworkDataset

def load_data(df,batchsize=8,aug=1):
    #call data loader
    if aug==1:
        data =SiameseNetworkDataset(df,image_D='2D',transform=(1,augmentation()))
    elif aug==2:
        data =SiameseNetworkDataset(df,image_D='2D',transform=(2,augmentation1()))
    elif aug==0:
        data =SiameseNetworkDataset(df,image_D='2D',transform=(1,no_augmentation()))
     
    #load images
    loader = DataLoader(data,shuffle=True,num_workers=0,batch_size=batchsize)
    return loader



#training
def train_dl(loader,epochs,model,device,criterion,opt):
    model=model.to(device)
    for _epoch in range(epochs):
        for batch in loader:
            img1,img2,label=batch
            img1_emb,img2_emb=model(img1.to(device)),model(img2.to(device))
            loss=criterion(img1_emb,img2_emb,label.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
    #print('model training completed')











