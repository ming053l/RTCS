#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import os
import glob
import time
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import json
from sklearn.metrics import log_loss
import pdb
import random as rn
from model import DFModel_var, DFModel_TSN
from utils import *
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from dataset import *
from scipy.io import savemat, loadmat
from math import acos, degrees


torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 16
device = 'cuda'

VAL_HR = 256
INTERVAL= 4
WIDTH=4
BANDS = 172
SIGMA = 0.0
TARGET_List  =['farm',  'city', 'lake', 'mou']

CR = 15
       
def loadTxt(fn):
    a = []
    with open(fn, 'r',encoding="utf-8") as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            a.append(fn)
    return a



def awgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/x.numel()
    npower = torch.sqrt(xpower / snr)
    return x + torch.randn(x.shape).cuda() * npower



# Loading all files in quene
GTs = {}
for TARGET in TARGET_List:
    ## Load test files from specified text with TARGET name (you can replace it with other path u want)
    valfn = loadTxt('data/val_%s.txt' % TARGET)  
    GTs[TARGET] = {}
    for fn in tqdm(valfn, total=len(valfn)):
        bn = os.path.basename(fn)
        GTs[TARGET][bn] = np.load('../HSDCR/'+fn).astype(np.float)
print("Data preparation done!")


# In[10]:

os.environ["CUDA_VISIBLE_DEVICES"]="3"
#device='cuda'
CR = 15
EP = 16150
SOURCE='lake'
TARGET='city'
#ckpt= '../%s/DCSN_%s_%s_cr_%d_epoch_%d.pth' % (SOURCE, SOURCE, TARGET, CR, EP)
ckpt = 'best/exp__lake_city_cr_15_best2.pth'
#state_dict = torch.load(ckpt, map_location=device)['dict']
state_dict = torch.load(ckpt, map_location=torch.device('cuda'))['dict']
## Model loading
model = DFModel_TSN(cr=CR, stream=3)  


from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

model.eval().to(device)


def psnr(x,y):
    bands = x.shape[2]
    x = np.reshape(x, [-1, bands])
    y = np.reshape(y, [-1, bands])
    msr = np.mean((x-y)**2, 0)
    maxval = np.max(y, 0)**2
    return np.mean(10*np.log10(maxval/msr))

def sam(x, y):
    num = np.sum(np.multiply(x, y), 2)
    den = np.sqrt(np.multiply(np.sum(x**2, 2), np.sum(y**2, 2)))
    sam = np.sum(np.degrees(np.arccos(num / den))) / (x.shape[1]*x.shape[0])
    return sam

def rmse(x, y):

    aux = (np.sum((y-x)**2, (0,1))) / (x.shape[0]*x.shape[1])
    r = np.sqrt(np.mean(aux))
    return r

def ERGAS(x, y, Resize_fact=4):

    err = y-x
    ergas=0
    for i in range(y.shape[-1]):
        ergas += np.mean(np.power(err[:,:,i],2)) / np.mean(y[:,:,i])**2
    ergas = (100.0/Resize_fact) * np.sqrt(1.0/y.shape[-1] * ergas)
    return ergas


index = {}
for TARGET in TARGET_List:
    index[TARGET]={}
    
for TARGET in TARGET_List:
    ## Load test files from specified text with TARGET name (you can replace it with other path u want)
    valfn = loadTxt('data/val_%s.txt' % TARGET)  

    ## Setup the dataloader
    val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root='../HSDCR/'), 
                                             batch_size=5, shuffle=False, pin_memory=True, drop_last=False)

   
    with torch.no_grad():
        rmses, sams, fnames, psnrs,ergass = [], [], [],[], []
        ep = 0
        
        for ind2, (vx, vfn) in tqdm(enumerate(val_loader), total=len(val_loader)):
            model.eval()
            vx = vx.view(vx.size()[0]*vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
            vx= vx.to(device).permute(0,3,1,2).float()
            start_time = time.time()
            val_dec,_,_,_,_ = model(vx)
            ep += (time.time()-start_time)
#             if SIGMA>0:
#                 val_dec = model(awgn(model(vx, mode=1), 30), mode=2)
#             else:
#                 val_dec,_ = model(vx)


            ## Recovery to image HSI
            val_batch_size = len(vfn)
            img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
            val_dec = val_dec.permute(0,2,3,1).cpu().numpy()
            cnt = 0

            for bt in range(val_batch_size):
                for z in range(0, VAL_HR, INTERVAL):
                    img[bt][:,z:z+WIDTH,:] = val_dec[cnt]
                    cnt +=1
                save_path = vfn[bt].split('/')
                save_path = save_path[-2] + '-' + save_path[-1]
#                 save_path = save_path[-1]
#                 np.save('Rec/%s.npy' % (save_path), img[bt])
                
#                 if 'Florida_65npy' in vfn[bt]:
#                     pdb.set_trace()
                    
    
                bn = os.path.basename(vfn[bt])
                GT = GTs[TARGET][bn]
                maxv, minv=np.max(GT), np.min(GT)
                img[bt] = img[bt]*(maxv-minv) + minv ## De-normalization
                savemat('Rec/%s.mat' % save_path, {'rec':img[bt]})
                
                sams.append(sam(GT,img[bt]))
                psnrs.append(psnr(GT,img[bt]))
                rmses.append(rmse(GT,img[bt]))
                ergass.append(ERGAS(GT,img[bt]))
                fnames.append(save_path)


        ep = ep / len(sams)
    
        print('%s in cr=%d: psnr/rmse/sam/ERGAS: %.3f / %.3f / %.3f /%.3f, AVG-Time: %.3f' %
              (TARGET, CR, np.mean(psnrs), np.mean(rmses), np.mean(sams), np.mean(ergass), ep))
        index[TARGET]['sam'] = sams
        index[TARGET]['psnr'] = psnrs
        index[TARGET]['rmse'] = rmses
        index[TARGET]['ergas'] = ergass


# In[10]:


s1, p1, r1, e1 = [],[],[],[]
for TARGET in TARGET_List:
    s1.append(np.mean(index[TARGET]['sam'] ))
    p1.append(np.mean(index[TARGET]['psnr']))
    r1.append(np.mean(index[TARGET]['rmse'] ))
    e1.append(np.mean(index[TARGET]['ergas'] ))
    
print('All PSNR/RMSR/SAM/ERGAS: %.3f/%.3f/%.3f/%.3f' % (np.mean(p1), np.mean(r1), np.mean(s1), np.mean(e1)))
print('time', ep)


# In[ ]:




