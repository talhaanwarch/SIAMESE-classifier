# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:15:37 2021

@author: TAC
"""


import random
from PIL import Image
import torch

class SiameseNetworkDataset():
    
    def __init__(self,imageFolderDataset,image_D='2D',transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform=transform
        self.image_D=image_D
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
            label=torch.zeros(1)
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break
            label=torch.ones(1)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        if self.image_D=='2D':
            img0=img0.convert("L")
            img1=img1.convert("L")
        
        if self.transform:
           img0=self.transform(img0) 
           img1=self.transform(img1) 

        
        return img0, img1 ,label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)