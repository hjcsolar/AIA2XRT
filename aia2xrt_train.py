# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:43:35 2018

@author: jkf
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import glob,os,random,pickle,sim
import sunpy.map
import numpy as np
from torch.utils import data
# from scipy.stats import pearsonr as pr
from scipy import stats

#from scipy.signal import medfilt2d 

from resnet import ResNet as NET 
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 1000 #最大迭代次数
learning_rate = 0.5*1e-4 #初始学习率
batchsize=10
IMsize=200 #训练图像的patch大小
scale=1 #如果采用qian放大，
inputCH=6
outputCH=1
convNUM=64
layers=20

curr_lr=learning_rate  #当前的学习率
K=0 #循环显示的初始参数

netfile='aia2Be_Thin64x20L2log1.mod' #模型文件名
parafile=netfile+'.train' #训练记录，主要是epoch，训练集和验证集的loss和学习率

class FitsSet(data.Dataset): #训练集数据
    def __init__(self,file):
        # 所有数据的绝对路径
        self.imgs=glob.glob(file)
    def __getitem__(self, index):
        img_path = self.imgs[index-1]
#        print(img_path)
        try:
            dat = sim.fitsread(img_path)[0] #根据数据文件格式打开文件 
            dat = np.maximum(dat,0)
            dat=np.log10(dat+1)
        except:
            print(img_path) 
        dat=sim.removenan(dat)
        # dat = dat[:,200:1024-200,200:1024-200]
        high=dat[6].astype('float32') #标签图像

        low=dat[:6].astype('float32') #输入图像
#        for i in range(3):
#            low[i+1]=low[i+1]/low[0]
#        high=high[100:-100,100:-100]
#        low=low[:,100:-100,100:-100]
       #随机数据增广
        flip=random.randint(0,1) #随机数据镜像
        if flip == 1:
            high=high[:,::-1]
            low=low[:,:,::-1]


        flip=random.randint(0,3) #随机数据旋转0，90，180，270度
        high=np.rot90(high,flip)
        low=np.rot90(low,flip,(1,2)) 
  

        _,h,w=low.shape
        #随机切大小有imsize的图像块
        imsize=np.min((IMsize,h,w))-3
        starth=0
        endh=h-imsize-1

        startw=0
        endw=w-imsize-1

        

        k=0
        while k<100:
            k+=1
            h0=random.randint(starth,endh)
            w0=random.randint(startw,endw)
            high0=high[h0*scale:(h0+imsize)*scale,w0*scale:(w0+imsize)*scale].astype('float32')
            low0=low[:,h0:h0+imsize,w0:w0+imsize].astype('float32')
            D=high0>1
            if D.sum()>50:
                break
        #输入图像和标签数据的亚像元对齐，注意：如果输入和标签图像不是一种图像，不能做这一步
#        [dx,dy,cor]=sim.xcorrcenter(low[0],HMI)
#
#
#        if np.abs(dx)>2 or np.abs(dy)>2:
#            high=sim.immove(high,dx,dy)
#            print(img_path,dx,dy,cor)

#        it=0
#        while abs(dx)>1 or abs(dy)>1:
#            it+=1
#            dx=abs(int(dx))
#            dy=abs(int(dy))
#            [dx,dy,cor]=sim.xcorrcenter(low[0,dx:-dx-1,dy:-dy-1],high[dx:-dx-1,dy:-dy-1])
#            high=sim.immove(high,dx,dy)
#            if it>10:
#                break
##
 
        high=np.ascontiguousarray(high0, dtype=np.float32) #数据规整到连续内存空间，这步如果不做，有时候会有问题
        low=np.ascontiguousarray(low0, dtype=np.float32)

        high=torch.from_numpy(high).unsqueeze(0) #对于单输入单输出的网络，要从二维图像，变为三维张量。其中第一维维图像通道数
#        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    def __len__(self):
        return len(self.imgs)
    
class validFitsSet(data.Dataset): #验证集数据，和训练集的区别在于不用做数据增广
    def __init__(self,file):
        # 所有图片的绝对路径
        self.imgs=glob.glob(file)
    def __getitem__(self, index):
        img_path = self.imgs[index-1]
        dat = sim.fitsread(img_path)[0] #根据数据文件格式打开文件  
        dat = np.maximum(dat,0)
        dat=np.log10(dat+1)
        high=dat[6].astype('float32') #标签图像

        low=dat[:6].astype('float32') #输入图像
#        for i in range(3):
#            low[i+1]=low[i+1]/low[0]

        high=high[400:600,400:600] #验证集图像大小，太大了内存会爆
        low=low[:,400:600,400:600]
#        
#        [dx,dy,cor]=sim.xcorrcenter(low[0],HMI)
#        high=sim.immove(high,dx,dy)       
#        it=0
#        while abs(dx)>1 or abs(dy)>1:
#            it+=1
#            dx=abs(int(dx))
#            dy=abs(int(dy))
#            [dx,dy,cor]=sim.xcorrcenter(low[dx:-dx-1,dy:-dy-1],high[dx:-dx-1,dy:-dy-1])
#            high=sim.immove(high,dx,dy)
#            if it>10:
#                break

        high=np.ascontiguousarray(high, dtype=np.float32)
        low=np.ascontiguousarray(low, dtype=np.float32)

        high=torch.from_numpy(high).unsqueeze(0)
#        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    def __len__(self):
        return len(self.imgs)
    
    
class myloss(nn.Module): #自定义损失函数。 根据不同情况，自己写
    def __init__(self):
        super(myloss,self).__init__()
        return
    def forward(self,w1,w2):#mse：最小平方误差函数
        T0=4000
#        arr2=(torch.abs(w1-w2)<T) *  (torch.abs(w2)<T0) *  (torch.abs(w2)>10)
        arr2=(torch.abs(w2)<T0) *(torch.abs(w2)>0)
 
        wg=torch.log(torch.abs(w2[arr2])+1)
        wgsum=torch.sum(wg)
#
        loss = torch.sum(wg*torch.pow(w1[arr2]-w2[arr2],2))/wgsum
#        loss = torch.mean(torch.abs(w1[arr2]-w2[arr2]))
        return loss



train_dataset=FitsSet('../train/*.fits') #训练集
valid_dataset=validFitsSet('../valid/*.fits') #验证集

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batchsize, 
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=2, 
                                           shuffle=False)
#载入历史训练参数或者初始化
try:
    para = pickle.load(open(parafile,'rb'))   
    epoch0=para[0]
    totloss=para[1]
    curr_lr=para[2]
except: 
    epoch0=0
    totloss=[]
    para=[epoch0, totloss, curr_lr]

#载入历史训练模型
try:
    model_state_dict = torch.load(netfile).state_dict()
    model = NET(inputCH,convNUM,layers,outputCH).to(device)
    model.load_state_dict(model_state_dict)
except:    
    model = NET(inputCH,convNUM,layers,outputCH).to(device)
    
model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练

#criterion = nn.L1Loss() #L1 损失函数
criterion= nn.MSELoss() #L2 损失函数
#criterion = myloss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #最优化算法

def update_lr(optimizer, lr):    #动态调整学习率函数
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader) #训练集批次（batch)数
valid_step=len(valid_loader) #验证集批次（batch)数

for epoch in range(epoch0,num_epochs):
    eploss=0
    
    torch.cuda.empty_cache() #清空GPU内存
    
    for i, (images, labels, name) in enumerate(train_loader): #训练一个batch,即一个step. images和 labels 分别对应数据集中的AIA 和XRT

        images = images.to(device) # 放入GPU
        labels = labels.to(device)
        

        outputs = model(images) #训练！
        wd=5
        loss= criterion(outputs[:,:,wd:-wd,wd:-wd].squeeze(), labels[:,:,wd:-wd,wd:-wd].squeeze()) #计算损失函数，由于卷积的边界问题，所以扣除了边上的数据，但这个边界宽度我也没有谱
        
        # Backward and optimize，反向递推梯度，修正权值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lossvalue=loss.item() #获取loss值
        eploss+=lossvalue #在一个step里面累积loss值

        if (i+1) % 10 == 0:    #N步显示一下
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} {:}" .format(epoch+1, num_epochs, i+1, total_step,lossvalue,name[0] ))

    Loss=[epoch,eploss/total_step]  #一个epoch的平均loss                                          

    if (epoch+1) % 10== 0: #多少epoch 显示一下图像和曲线，以及保存模型和参数

        pre = outputs.cpu().detach().numpy()[:,:,wd:-wd,wd:-wd] #这个epoch的最后一个step(batch)的预测值
        predicted=pre[0,0] #这个batch中的第一张图
        y_t=labels.cpu().detach().numpy()[:,:,wd:-wd,wd:-wd] #这个epoch的最后一个step(batch)的标签值
        y_train=y_t[0,0]
        
        name= name[0]#文件

        AIA=images.cpu().detach().numpy()[0,0:6] #原始的AIA 图
        AIA=sim.zscore2(AIA[:,wd:-wd,wd:-wd])
        
        mi=y_t.min() #标签值得最大最小值
        mx=y_t.max()
        
        #计算某个范围内标签值和预测值的线性拟合系数和相关系数
        # arr=(np.abs(y_train)<5000)*(np.abs(y_train)>0)
        # S1=np.polyfit(y_train[arr].flatten(),predicted[arr].flatten(),1)
        # P1=np.poly1d(S1) #拟合的斜率
        # pr1=pr(y_train.flatten(),predicted.flatten()) #相关系数
        slope,intercept,r_value,p_value,std_err=stats.linregress(x=y_t.flatten(),y=pre.flatten())
        pr1=r_value
        P1=[intercept,slope]
        
        delta=(predicted-y_train) #预测值和标签的残差图
        re=delta.std() #残差RMS
        ssim=compare_ssim(sim.imnorm(predicted),sim.imnorm(y_train))
#        ssim=compare_ssim(np.uint8(sim.imnorm(predicted)*255),np.uint8(sim.imnorm(y_train)*255)) #计算SSIM
        MXRT=sim.zscore2(predicted)
        RXRT=sim.zscore2(y_train)
        DXRT=sim.zscore2(delta)
        XRT=np.stack((MXRT,RXRT,DXRT))
        
            
        fig=plt.figure(num=1,figsize=(8,7))
        namesub=os.path.basename(name)
        fig.suptitle('Epoch:'+str(epoch+1)+'  Lr:'+str(curr_lr)+
                              '  SSIM:'+str(ssim)[:6]+'  Resid:'+str(re)[:7]+
                              '  '+str(namesub[13:50]),fontsize=12)
        #图像窗口，low,预测，标签，残差
        wave=['171','193','211','335','131','94']
        cmap=['sdoaia171','sdoaia193','sdoaia211',
              'sdoaia335','sdoaia131','sdoaia94']
        for i in range(0,6) :
            ax=plt.subplot(361+i)
            ax.tick_params(labelsize=4)
            dis=ax.imshow(AIA[i,:,:],vmax=3,vmin=-2,cmap=cmap[i], interpolation='bicubic')   
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(wave[i],fontsize=9,y=-0.3)
            plt.subplots_adjust(wspace=0,hspace=0)
            plt.pause(0.1)
            plt.draw()
      
        title_XRT=['Model XRT','Real XRT','Difference']
        color_XRT=['hinodexrt','hinodexrt','gray']
        for i in range(0,3):
            ax1=plt.subplot(334+i)
            ax1.tick_params(labelsize=5)
            dis=ax1.imshow(XRT[i,:,:],vmax=5,vmin=-3,cmap=color_XRT[i], interpolation='bicubic') 
            plt.title(title_XRT[i],fontsize=10,y=1)
            plt.pause(0.1)
            plt.draw() 


        #标签和预测值得45度线
        ax2=plt.subplot(325)
        plt.cla()
        #ax2.plot(y_t.flatten(),pre.flatten(),'.r',MarkerSize=1)
        z=ax2.hist2d(x=y_t.flatten(),y=pre.flatten(),bins=200,
                     norm=mpl.colors.LogNorm())
        
        plt.xlim((mi,mx))
        plt.ylim((mi,mx))
        
        
        plt.plot([mi,mx],[mi,mx],'-g')
        plt.plot([mi,mx],[slope*mi+intercept,slope*mx+intercept],'--r')
        plt.tight_layout()
        title='slope:'+str(P1[1])[:5]+' corr:'+str(pr1)[:5]
        plt.title(title,fontsize=9)
        torch.cuda.empty_cache()   

        #计算验证集loss 
        vloss=0
        for i, (images, labels,name) in enumerate(valid_loader):
            
            images = images.to(device)
            labels = labels.to(device)
              
            outputs = model(images)
    
            loss_valid= criterion(outputs[:,:,wd:-wd,wd:-wd].squeeze(), labels[:,:,wd:-wd,wd:-wd].squeeze())
            vloss+=loss_valid.item()
            images=images.cpu() #把数据返回CPU，便于清空GPU内存
            labels=labels.cpu()
            outputs=outputs.cpu()
            
        vloss/=valid_step #平均的验证集loss     
        Loss.append(vloss)       
        totloss.append(Loss) 

       #显示损失曲线
        ax3=plt.subplot(326)
        plt.cla()

        tloss=np.array(totloss)
        plt.plot(tloss[0:,0],tloss[0:,1],'r',MarkerSize=1) #训练集loss
        plt.plot(tloss[0:,0],tloss[0:,2],'b',MarkerSize=1) #验证集loss

        plt.title('Train_loss:'+str(Loss[1])[:7]+' Vallid_loss:'+str(Loss[2])[:7],fontsize=9)
        plt.ylim((0.0,0.1)) #loss 曲线的显示范围，可按照实际调整
        plt.tight_layout()
        plt.pause(0.1)
        plt.draw() 
        plt.clf()
        #保存模型和参数
        torch.save(model, netfile)
        # torch.save(model,'./model/'+str(epoch)+netfile)
        para[0]=epoch
        para[1]=(totloss)
        para[2]=(curr_lr)
        pickle.dump(para, open(parafile, 'wb'))
    
    # 动态改变学习率 learning rate
    if (epoch+1) % 1000 == 0:
        curr_lr /= 1.1
        update_lr(optimizer, curr_lr)
