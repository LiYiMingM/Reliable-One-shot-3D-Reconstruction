import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
import os
import torchvision
from torchvision.utils import save_image
import time
'Computes and stores the average and current value.'
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

' network training function '
def train_net(net, device, loader, optimizer, loss_l1,loss_l2,batch_size):
    net.train()
    
    train_loss_unwrap = AverageMeter() 
    a1_ave_train_unwrap =AverageMeter() 
    a2_ave_train_unwrap=AverageMeter() 
    a3_ave_train_unwrap=AverageMeter() 
    train_l1loss_unwrap=AverageMeter() 
    train_l2loss_unwrap=AverageMeter() 
    train_silogloss_unwrap= AverageMeter()   
    for batch_idx, (input,gt_unwrap ) in enumerate(loader):

        input,gt_unwrap = input.to(device),gt_unwrap.to(device)                                                 # Send data to GPU

        output_train_unwrap = net(input) # Forward
    

        loss_unwrap_l1=loss_l1(output_train_unwrap, gt_unwrap)
        loss_unwrap_l2=torch.sqrt(loss_l2(output_train_unwrap, gt_unwrap)) 
   
        
        loss=loss_unwrap_l1*1+loss_unwrap_l2*0.5


        loss.requires_grad_(True)
        train_loss_unwrap.update(loss.item(), output_train_unwrap.size(0)) # Update the record

        train_l1loss_unwrap.update(loss_unwrap_l1.item(), output_train_unwrap.size(0)) # Update the record
        train_l2loss_unwrap.update(loss_unwrap_l2.item(), output_train_unwrap.size(0)) # Update the record

        thresh = torch.maximum((gt_unwrap/output_train_unwrap), (output_train_unwrap/gt_unwrap))#thresh[batchsize,1,512,640]

        a1 = (thresh < 1.25).to(torch.float32).mean()#
        a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
        a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
        a1_ave_train_unwrap.update(a1.item(),output_train_unwrap.size(0)) #output_val1.size(0)=batchsize
        a2_ave_train_unwrap.update(a2.item(),output_train_unwrap.size(0)) 
        a3_ave_train_unwrap.update(a3.item(),output_train_unwrap.size(0)) 



        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    output_train_unwrap=output_train_unwrap[-1]
    output_train_unwrap=output_train_unwrap/ (torch.max(output_train_unwrap) - torch.min(output_train_unwrap))

        
    print(' train_Loss_total: ' + str(round(train_loss_unwrap.avg, 6)))   
    print(' train_L1loss_unwrap: ' + str(round(train_l1loss_unwrap.avg, 6)))
    print(' train_L2loss_unwrap: ' + str(round(train_l2loss_unwrap.avg, 6)))
    # print(' train_silogloss_unwrap: ' + str(round(train_silogloss_unwrap.avg, 6)))
    return train_loss_unwrap.avg,output_train_unwrap,\
           a1_ave_train_unwrap.avg,a2_ave_train_unwrap.avg,a3_ave_train_unwrap.avg,\
           train_l1loss_unwrap.avg,train_l2loss_unwrap.avg

' network validating function '
def val_net(net, device, loader,loss_l1,loss_l2,batch_size):

    net.eval()
    val_loss_unwrap= AverageMeter() 
    a1_ave_val_unwrap = AverageMeter()
    a2_ave_val_unwrap = AverageMeter()
    a3_ave_val_unwrap = AverageMeter()   
    val_l1loss_unwrap=AverageMeter() 
    val_l2loss_unwrap=AverageMeter()    
    val_silogloss_unwrap=AverageMeter()    
    with torch.no_grad():
        for batch_idx, (input,gt_unwrap ) in enumerate(loader):
            input,gt_unwrap = input.to(device),gt_unwrap.to(device)                                               # Send data to GPU
            output_val_unwrap = net(input) # Forward
            output_image = output_val_unwrap.squeeze(0).squeeze(0)  


            l1loss_unwrap_val=loss_l1(output_val_unwrap, gt_unwrap)
            l2loss_unwrap_val=torch.sqrt(loss_l2(output_val_unwrap, gt_unwrap) )
 
            loss=l1loss_unwrap_val*1+l2loss_unwrap_val*0.5

            val_loss_unwrap.update(loss.item(), output_val_unwrap.size(0)) # Update the record

            val_l1loss_unwrap.update(l1loss_unwrap_val.item(), output_val_unwrap.size(0)) # Update the record
            val_l2loss_unwrap.update(l2loss_unwrap_val.item(), output_val_unwrap.size(0)) # Update the record

            thresh = torch.maximum((gt_unwrap/output_val_unwrap), (output_val_unwrap/ gt_unwrap))#thresh[batchsize,1,512,640]

            a1 = (thresh < 1.25).to(torch.float32).mean()#a1 #[batchsize,1,512,640]#
            a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
            a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
            a1_ave_val_unwrap.update(a1.item(),output_val_unwrap.size(0)) #output_val1.size(0)=batchsize
            a2_ave_val_unwrap.update(a2.item(),output_val_unwrap.size(0)) 
            a3_ave_val_unwrap.update(a3.item(),output_val_unwrap.size(0)) 



    output_val_unwrap=output_val_unwrap[-1]
    output_val_unwrap=output_val_unwrap/ (torch.max(output_val_unwrap) - torch.min(output_val_unwrap))


    print(' Val_L1loss_unwrap: ' + str(round(val_l1loss_unwrap.avg, 6)))
    print(' Val_L2loss_unwrap: ' + str(round(val_l2loss_unwrap.avg, 6)))
    print(' a1_ave_val: ' + str(round(a1_ave_val_unwrap.avg, 4)))   
    print(' a2_ave_val: ' + str(round(a2_ave_val_unwrap.avg, 4)))  
    print(' a3_ave_val: ' + str(round(a3_ave_val_unwrap.avg, 4)))  
    return val_loss_unwrap.avg ,output_val_unwrap,\
    a1_ave_val_unwrap.avg,a2_ave_val_unwrap.avg,a3_ave_val_unwrap.avg,\
    val_l1loss_unwrap.avg,val_l2loss_unwrap.avg


