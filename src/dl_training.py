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
import random
def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


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

def al_augmentation():
    """albumentations image augmentation"""
    return Compose([
            RandomResizedCrop(248, 248),
            #Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            GaussNoise() ,
            #IAASuperpixels(),
            Normalize(mean=[0.2,], std=[0.2,], p=1.0),
            #CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  



#image augmentation
from torchvision import transforms
def torch_augmentation():
    aug=transforms.Compose([
        transforms.Resize(size=(248,248)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
#         transforms.CenterCrop(248),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.2, ), (0.2, )),
                          ])
    return aug




def no_augmentation():
    aug=transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.2, ), (0.2, )),
                          ])
    return aug

from torch.utils.data.dataloader import DataLoader
from dataloader import SiameseNetworkDataset

def load_data(df,batchsize=8,aug=1,image_D='gray'):
    #call data loader
    if aug==1:#torch
        data =SiameseNetworkDataset(df,image_D=image_D,transform=(1,torch_augmentation()))
    elif aug==2:#albumentation
        data =SiameseNetworkDataset(df,image_D=image_D,transform=(2,al_augmentation()))
    elif aug==0:#no augmentation
        data =SiameseNetworkDataset(df,image_D=image_D,transform=(1,no_augmentation()))
#     elif aug==3:#random augmentation
#         data =SiameseNetworkDataset(df,image_D=image_D,transform=(3,rand_augmentation()))
     
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











