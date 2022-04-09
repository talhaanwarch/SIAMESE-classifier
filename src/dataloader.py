# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:15:37 2021

@author: TAC
"""


import random
from PIL import Image
import torch
import numpy as np
class SiameseNetworkDataset():
    
    def __init__(self,df,image_D='gray',transform=None):
        self.df = df    
        self.transform=transform
        self.image_D=image_D
        
    def __getitem__(self,index):
        img0 = random.choice(self.df)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1 = random.choice(self.df) 
                if img0[1]==img1[1]:
                    break
            label=torch.zeros(1)
        else:
            while True:
                #keep looping till a different class image is found
                
                img1 = random.choice(self.df) 
                if img0[1] !=img1[1]:
                    break
            label=torch.ones(1)
        img0 = Image.open(img0[0])
        img1 = Image.open(img1[0])
        
        if self.image_D=='gray':
            img0=img0.convert("L")
            img1=img1.convert("L")
        #print(img0.dtype)
  
        if self.transform[0]==2:
            img0 = self.transform[1](image=np.array(img0))['image']   
            img1 = self.transform[1](image=np.array(img1))['image']  
        else:
            img0=self.transform[1](img0) 
            img1=self.transform[1](img1) 

        
        return img0, img1 ,label
    
    def __len__(self):
        return len(self.df)