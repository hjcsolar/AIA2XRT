# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:43:35 2018

@author: jkf
"""
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import os,sim


import numpy as np

from scipy.stats import pearsonr as pr
from scipy import stats
#from scipy.signal import medfilt2d 

import sunpy.map
from sunpy.time import parse_time
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 
from resnet1 import ResNet as NET 
netfile='../CNNmodel/aia2Be_Thin64x20L2log1.mod' #模型文件名
file='../data_test/heap_com_XRT_LMS_20131013_180218_Be_thin_Open.fits' #测试fits
# from resnet1x1 import ResNet as NET 
# netfile='Be_thin_aia2xrt_CNN1x1.mod' #模型文件名

# Hyper-parameters

scale=1 #如果采用后放大，
inputCH=6
outputCH=1
convNUM=64
layers=20
cmap=['sdoaia171','sdoaia193','sdoaia211', 'sdoaia335','sdoaia131','sdoaia94','hinodexrt']


def validFitsSet(img_path): #验证集数据，和训练集的区别在于不用做数据增广

        dat = sim.fitsread(img_path)[0] #根据数据文件格式打开文件   
        dat = np.maximum(dat,0)
        # dat=sim.removenan(dat)
        dat=sim.removenan(np.log10(dat+1))
        # dat=dat[:,::-1,:]
        # dat = dat[:,200:400,250:540]
        high=dat[6].astype('float32') #标签图像
        # high=high+np.log10(4)
        low=dat[:6].astype('float32') #输入图像
        
      
        high=np.ascontiguousarray(high, dtype=np.float32)
        low=np.ascontiguousarray(low, dtype=np.float32)


        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    
images,y_train,name = validFitsSet(file) #验证集


try:
    model_state_dict = torch.load(netfile,map_location=device).state_dict()
    model = NET(inputCH,convNUM,layers,outputCH).to(device)
    model.load_state_dict(model_state_dict)
except:    
   print('Cannot find the model')
    
model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练
imsize=y_train.shape[0]
frame=10 #必须偶数
overlap=frame//2
block=4
wd=(imsize-frame)//block
#wd=250

predicted=np.zeros((imsize,imsize))
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

predict=(10**predicted)-1  #de-log10
y_t=(10**y_train)-1
# predicted=predicted[20:imsize-20,20:imsize-20]
# y_train=y_train[20:imsize-20,20:imsize-20]

# predicted=predicted[250:550,150:450]
# y_train=y_train[250:550,150:450]

# predicted_log=np.log10(predicted+5)
# y_train_log=np.log10(y_train+5)
mi=y_train.min() #标签值得最大最小值
mx=y_train.max()

#计算某个范围内标签值和预测值的线性拟合系数和相关系数
#arr=(np.abs(y_train)<5000)*(np.abs(y_train)>0)
S1=np.polyfit(y_train[frame:-frame,frame:-frame].flatten(),predicted[frame:-frame,frame:-frame].flatten(),1)
P1=np.poly1d(S1) #拟合的斜率
pr1=pr(y_train[frame:-frame,frame:-frame].flatten(),predicted[frame:-frame,frame:-frame].flatten()) #相关系数



delta=(predicted-y_train) #预测值和标签的残差图
re=delta[frame:-frame,frame:-frame].std() #残差RMS

ssim=compare_ssim(sim.imnorm(predicted[frame:-frame,frame:-frame]),sim.imnorm(y_train[frame:-frame,frame:-frame])) #计算SSIM
#拼接图像
mtmp=np.hstack((predicted,y_train))     

fig=plt.figure(num=netfile,figsize=(8,7))
ax=plt.subplot(211)

plt.imshow(mtmp,cmap=cmap[6],vmin=0,vmax=3.5,interpolation='bicubic',origin='lower')
plt.colorbar()  
plt.axis('off')   
plt.pause(0.1)
plt.draw()
namesub=os.path.basename(name)
# plt.title('Left:AI-Generated  '+'  SSIM:'+str(ssim)[:6]+'   Right:Obsevered')
plt.title('Left:AI-Predicted  '+'         '+'   Right:Obsevered')
id=namesub.find('_20')+1
time=parse_time(namesub[id:id+15]).iso
# plt.text(0.5,0.05,time[0:19],fontsize=6,weight='bold',ha='left',color='white',transform=ax.transAxes)
# ax.annotate("",xy=(140,840),xytext=(250,863),arrowprops=dict(arrowstyle="->",color='white'))
# ax.annotate("",xy=(140+imsize-40,840),xytext=(250+imsize-40,863),arrowprops=dict(arrowstyle="->",color='white'))
rect1 = plt.Rectangle((306,566),width=236,height=309,edgecolor='white',linewidth=0.5,fill=False)
rect2 = plt.Rectangle((306+imsize,566),width=236,height=309,edgecolor='white',linewidth=0.5,fill=False)
ax.add_patch(rect1)
ax.add_patch(rect2)

#标签和预测值得45度线
ax=plt.subplot(223)
plt.imshow(delta,vmin=-1,vmax=1,origin='lower',cmap='gray')
plt.axis('off')
cbar=plt.colorbar()
cbar.set_ticks([-1,0,1])
plt.title('Difference')
# ax.annotate("",xy=(140,840),xytext=(250,863),arrowprops=dict(arrowstyle="->",color='white'))


ax1=plt.subplot(224)
z=plt.hist2d(x=y_train[frame:-frame,frame:-frame].flatten(),y=predicted[frame:-frame,frame:-frame].flatten(),norm=mpl.colors.LogNorm(),bins=100,range=[[mi,mx],[mi,mx]])
cb=plt.colorbar()
cb.set_label('Numbers of pixels')
plt.xlim((mi,mx))
plt.ylim((mi,mx))

id1=namesub.find('S_')+2
id2=namesub.find('.fits')
plt.suptitle(namesub[id1:id2],fontsize=15,weight='normal',y=0.96)
plt.plot([mi,mx],[mi,mx],'-g')
title='slope:'+str(P1[1])[:5]+' corr:'+str(pr1[0])[:5]
plt.xlabel('Observed [log10 DN/s/pixel]')
plt.ylabel('Predicted [log10 DN/s/pixel]')
plt.title('2D histgram')

# plt.plot([mi,mx],[P1[1]*mi+P1[0],P1[1]*mx+P1[0]],'--c')
# plt.text(0.5,0.1,'Correlation:'+str(pr1[0])[:5],fontsize=9,transform=ax1.transAxes)
# plt.text(0.01,0.9,'Linear fit:'+'$y$'+'='+str(P1[1])[:5]+'$x$'+'$+$'+str(P1[0])[:5],fontsize=9,transform=ax1.transAxes,color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x=y_train[frame:-frame,frame:-frame].flatten(),y=predicted[frame:-frame,frame:-frame].flatten())
plt.plot([mi,mx],[slope*mi+intercept,slope*mx+intercept],'--r')
plt.text(0.01,0.92,'Linear fit:'+'$y$'+'='+str(slope)[:5]+'$x$'+'$+$'+str(intercept)[:5],fontsize=9,transform=ax1.transAxes,color='red')
plt.text(0.6,0.1,'CC:'+str(r_value)[:5],fontsize=9,transform=ax1.transAxes,color='blue')

id=file.find('_L')+1
name1='xrt1_'+file[id:id+50]
# sim.fitswrite(name1,predict)