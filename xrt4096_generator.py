#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:34:33 2020

@author: hongjunchao
"""

from matplotlib import pyplot as plt
import torch
import os,sim
import numpy as np
import sunpy.map
import glob
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from resnet1 import ResNet as NET 

# netfile='aia2Be_Thin64x20L2log1.mod' #模型文件名
# xrtchannel='Be_thin_Open'



# netfile='aia2Al_mesh_64x20L2log1.mod'
# xrtchannel='Open_Al_mesh'

netfile='../CNNmodel/aia2Ti_poly64x20L2log1.mod'
xrtchannel='Open_Ti_poly'


dir='./'


# Hyper-parameters

scale=1 #如果采用后放大，
inputCH=6
outputCH=1
convNUM=64
layers=20
cmap=['sdoaia171','sdoaia193','sdoaia211', 'sdoaia335','sdoaia131','sdoaia94','hinodexrt']


def FitsSet(img_path): 

        dat,hdr = sim.fitsread(img_path)  
        dat = np.maximum(dat,0)
        dat=sim.removenan(np.log10(dat+1))
        low=dat[:6].astype('float32') #输入图像
        low=np.ascontiguousarray(low, dtype=np.float32)


        low=torch.from_numpy(low).unsqueeze(0)

        return low,hdr,img_path



try:
    model_state_dict = torch.load(netfile,map_location=device).state_dict()
    model = NET(inputCH,convNUM,layers,outputCH).to(device)
    model.load_state_dict(model_state_dict)
except:    
   print('Cannot find the model')
    
model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练
imsize=4096
frame=10 #必须偶数
overlap=frame//2
block=4
wd=(imsize-frame)//block
predicted=np.zeros((imsize,imsize))
aiapix=0.6
xrtpix=2.05


file=dir+'heap6_*.fits'    # AIA input
faia=sorted(glob.glob(file))
n=np.max(np.arange(0,len(faia)))
j=-1
for i in faia :
    j=j+1
    # if j != 38 :
        # continue
    images,head,name = FitsSet(i) #验证集
    for row in range(block): 
        print(row)
        for col in range(block):
            image=images[:,:,col*wd:(col+1)*wd+frame,row*wd:wd*(row+1)+frame]
            image=image.to(device)
        #    image = images[:,:,i*col:(i+1)*col,:].to(device) # 放入GPU
        #    outputs = model(image) 
        #    predicted[i*col:(i+1)*col,:] = outputs.cpu().detach().numpy().squeeze()#这个epoch的最后一个step(batch)的预测值
            pre = model(image).cpu().detach().numpy().squeeze()
            predicted[overlap+col*wd:overlap+(col+1)*wd,overlap+row*wd:overlap+(row+1)*wd]=pre[overlap:-overlap,overlap:-overlap]


    #from 1024x1024XRT pixel-flux level  to 4096x4096AIA-XRT pixel-flux level
    predicted=predicted-np.log10((xrtpix/aiapix)**2.)   
    predict=10**predicted-1  #de-log10

    fig=plt.figure(num=netfile,figsize=(8,7))
    plt.subplot(211)

    plt.imshow(predicted,cmap=cmap[6], vmin=-3,vmax=5,interpolation='bicubic')
    plt.colorbar()     
    plt.pause(0.1)
    plt.draw()
    # if n !=0 or j !=n :
        # plt.clf()
    namesub=os.path.basename(name)

    id=namesub.find('_20')+1
    name1='xrt1_aia_'+namesub[id:id+15]+'_'+xrtchannel+'.fits'
    head['filename']=name1
    head['origin']='XRT_AIA_'+xrtchannel

    sim.fitswrite(dir+name1,predict,header=head)
    print(dir+name1)

