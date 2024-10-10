import torch
import torch.nn as nn
import time
from tqdm import tqdm
from optparse import OptionParser
import os
import csv
import numpy as np

import sys

from resnet_unet import WHOLE_NET
from dataset_read import get_dataloaders
from train_func import train_net, val_net
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from PIL import Image
from thop import profile
from thop import clever_format
import random
from datetime import datetime




# os.environ{'WANDB_API_KEY'}='887a7c15aacb5cf7e16c8e1afa5040fdff712863'
import wandb
wandb.login()
##################################################

###################################################
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=8, type='int', help='batch size')
    parser.add_option('-l', '--learning_rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('-v', '--val_percentage', dest='val_perc', default=0.1 ,type='float', help='validation percentage')


    parser.add_option('-r', '--root', dest='root', default=".../statue_1605/", help='root directory')
    parser.add_option('-i', '--input1', dest='input1', default='grating_4', help='folder of input')
    parser.add_option('-g', '--ground_truth1', dest='gt_unwrap', default='unwrapped_phase', help='folder of ground truth')
    parser.add_option('-n', '--ground_truth2', dest='gt_phl3', default='phl_3', help='folder of ground truth')
    parser.add_option('-k', '--ground_truth3', dest='gt_wrap', default='wrapped_phase_4', help='folder of ground truth')

    parser.add_option( '--w1', dest='w1', default=1, type='float',help='folder of ground truth')
    parser.add_option( '--w2', dest='w2', default=1, type='float',help='folder of ground truth')
    parser.add_option( '--w3', dest='w3', default=100, type='float',help='folder of ground truth')

    parser.add_option('-m', '--model', dest='model', default='Result/...', help='folder for model/weights')
    parser.add_option('--wandb_project_name',dest='wandb_project_name',default='statue_1605')  
    parser.add_option('-p', '--model_pre', dest='model_pre', default='.../weights.pth', help='pre_train_weights')  
    (options, args) = parser.parse_args()
    return options

' Run of the training and validation '
def setup_and_run_train(load_weights,dir_input1,dir_gt_phl3, dir_gt_wrap, dir_gt_unwrap, dir_model,wandb_project_name, val_perc, batch_size, epochs,lr,log_name1,w1,w2,w3,test_size=0.1):
    
    time_start = time.time()
  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device_ids = [0,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
    net = WHOLE_NET(backbone='resnet34')
   
    net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()

    # net = net.to(device)
    net.train()
 
    # Load the dataset
    train_loader, val_loader, _ = get_dataloaders(dir_input1,dir_gt_phl3,\
                                                  dir_gt_wrap, dir_gt_unwrap, batch_size,val_perc,test_size)
    # Definition of the optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    # Definition of the loss function
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    # Set the header for csv
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-06, last_epoch=-1)
    log_name=f"{log_name1.replace('/', '_')}_{w1}-{w2}-{w3}"
    print(log_name)
    folder_path=log_name1+'wandb'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    wandb.init(project=wandb_project_name,dir=folder_path,
               mode="offline",
           config={
                    "project_name":wandb_project_name,#
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epoch": epochs,
                    "network": net},
              )
    wandb.run.name = f"{log_name}"
    wandb.watch(net)



    header = ['epoch', 'learning rate', 'train loss_total','val loss_total', 'time cost now/second',
        'train loss_unwrap','val loss_unwrap', 'train loss_phl3','val loss_phl3', 'train loss_wrap','val loss_wrap' ,
          'l1loss_train_unwrap','l2loss_train_unwrap','l1loss_val_unwrap','l2loss_val_unwrap']
    best_loss = 1000000
    


    # Ready to use the tqdm (A Fast, Extensible Progress Bar for Python and CLI)
    for epoch in tqdm(range(epochs)):

        print('\ Learning rate = ' , round(scheduler.get_last_lr()[0],10), end= ' ')
        # Get training loss function and validating loss function

        train_loss_phl3,output_train_phl3,train_loss_wrap,\
        output_train_wrap,train_loss_unwrap,output_train_unwrap,train_loss_total,l1loss_train_unwrap,l2loss_train_unwrap= \
        train_net(net, device, train_loader, optimizer, loss_l1,loss_l2,w1,w2,w3,batch_size)

        scheduler.step()
        val_loss_phl3,output_val_phl3,val_loss_wrap,\
        output_val_wrap,val_loss_unwrap,output_val_unwrap,val_loss_total,l1loss_val_unwrap,l2loss_val_unwrap= \
        val_net(net, device, val_loader, loss_l1,loss_l2,w1,w2,w3,batch_size)



        output_train_unwrap = wandb.Image(output_train_unwrap, caption="epoch:{}".format(epoch))  # attention!!!
        output_train_wrap = wandb.Image(output_train_wrap, caption="epoch:{}".format(epoch))  # attention!!!
        output_train_phl3= wandb.Image(output_train_phl3, caption="epoch:{}".format(epoch))  # attention!!!

        wandb.log({"output_train_unwrap": output_train_unwrap,'epoch':epoch})
        wandb.log({"output_train_wrap": output_train_wrap,'epoch':epoch})
        wandb.log({"output_train_phl3": output_train_phl3,'epoch':epoch})
  
    
        wandb.log({'train_loss_total': train_loss_total,"epoch":epoch}) 
        wandb.log({'train_l1+l2loss_unwrap': train_loss_unwrap,"epoch":epoch})
        wandb.log({'train_l1loss_unwrap': l1loss_train_unwrap,"epoch":epoch})   
        wandb.log({'train_l2loss_unwrap': l2loss_train_unwrap,"epoch":epoch})                
        wandb.log({'train_loss_wrap': train_loss_wrap,"epoch":epoch})   
        wandb.log({'train_loss_phl3': train_loss_phl3,"epoch":epoch})   
                
   


               
        output_val_unwrap = wandb.Image(output_val_unwrap, caption="epoch:{}".format(epoch))  # attention!!!
        output_val_wrap = wandb.Image(output_val_wrap, caption="epoch:{}".format(epoch))  # attention!!!
        output_val_phl3= wandb.Image(output_val_phl3, caption="epoch:{}".format(epoch))  # attention!!!

        wandb.log({"output_val_unwrap": output_val_unwrap,'epoch':epoch})
        wandb.log({"output_val_wrap": output_val_wrap,'epoch':epoch})
        wandb.log({"output_val_phl3": output_val_phl3,'epoch':epoch})
 
        
        wandb.log({'val_loss_total': val_loss_total,"epoch":epoch}) 
        wandb.log({'val_loss_unwrap': val_loss_unwrap,"epoch":epoch})
        wandb.log({'val_l1loss_unwrap': l1loss_val_unwrap,"epoch":epoch})
        wandb.log({'val_l22oss_unwrap': l2loss_val_unwrap,"epoch":epoch})
        wandb.log({'val_loss_wrap': val_loss_wrap,"epoch":epoch})   
        wandb.log({'val_loss_phl3': val_loss_phl3,"epoch":epoch})   
                  

            
        # Get time cost now
        time_cost_now = time.time() - time_start
        # Set the values for csv
    
        values = [epoch+1, round(scheduler.get_last_lr()[0],10), '%.6f' % train_loss_total, '%.6f' % val_loss_total,\
                 '%.6f' % time_cost_now,\
                  '%.6f' % train_loss_unwrap,'%.6f' % val_loss_unwrap,'%.6f' % train_loss_phl3,'%.6f' % val_loss_phl3,\
                  '%.6f' % train_loss_wrap,'%.6f' % val_loss_wrap,\
                  '%.6f' % l1loss_train_unwrap,'%.6f' % l2loss_train_unwrap,'%.6f' % l1loss_val_unwrap,'%.6f' % l2loss_val_unwrap]        
# Save epoch, learning rate, train loss, val loss and time cost now to a csv
        if not os.path.exists(args.root + args.model + '/', ):
            os.makedirs(args.root + args.model + '/', )
        path_csv = dir_model +  f"trainloss_single---{w1}-{w2}-{w3}.csv"
        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer_csv = csv.writer(file)
            writer_csv.writerow(header)
            writer_csv.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer_csv = csv.writer(file)
            writer_csv.writerow(values)
        file.close()

        best_model_info = {
        'epoch': None,
        'train_loss_unwrap': None,
        'val_loss_unwrap': None,
        'l1loss_train_unwrap': None,
        'l2loss_train_unwrap': None,
        'l1loss_val_unwrap': None,
        'l2loss_val_unwrap': None,

        'state_dict': None,
        'optimizer_state_dict': None,
        'dir_model':None
        }
# Save model
        if l1loss_val_unwrap < best_loss:
            best_loss = l1loss_val_unwrap
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'train_loss': train_loss_unwrap,
                    'val_loss': val_loss_unwrap,
                    'train_loss_total': train_loss_total,
                    'l1loss_val_unwrap': l1loss_val_unwrap,
                    'optimizer' : optimizer.state_dict(),
                }, dir_model + f"weights_single-{w1}-{w2}-{w3}.pth")
         
            best_model_info.update({
                    'epoch': epoch + 1,
                    'train_loss_unwrap': train_loss_unwrap,
                    'val_loss_unwrap': val_loss_unwrap,
                    'l1loss_train_unwrap': l1loss_train_unwrap,
                    'l2loss_train_unwrap': l2loss_train_unwrap,
                    'l1loss_val_unwrap': l1loss_val_unwrap,
                    'l2loss_val_unwrap': l2loss_val_unwrap,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dir_model':dir_model
                })
         
            # print('1#############1train_loss_unwrap_w',best_model_info['a2_ave_val_unwrap'])
            # print('1##############1Type of train_loss_unwrap_w:', type(best_model_info['a2_ave_val_unwrap']))

    # Save the weights
    # wandb.save("weights.pth")       
    time_all = time.time() - time_start
#    print("Total time %.4f seconds for training" % (time_all))
    # log_writer.close()
    wandb.finish()

    return best_model_info

' Run the application '
if __name__ == "__main__":

    setup_seed(20)
    args = get_args()
    print("w1",args.w1,"w2",args.w2,"w3",args.w3)
    print('batch_size',args.batch_size)
    epoch_best, train_loss_unwrap_best, val_loss_unwrap_best = 1000000, 1000000, 1000000
    result_all = []

    time_start = time.time()

    best_model_info_1=setup_and_run_train(
            load_weights = args.root + args.model_pre,
            dir_input1=args.root + args.input1 + '/',
            dir_gt_unwrap = args.root +args.gt_unwrap+'/',
            dir_gt_phl3 = args.root +args.gt_phl3+'/',
            dir_gt_wrap = args.root +args.gt_wrap+'/',
            dir_model=args.root + args.model + '/',
            val_perc = args.val_perc,
            batch_size = args.batch_size,
            epochs = args.epochs,
            wandb_project_name=args.wandb_project_name,            
            lr = args.lr,
            log_name1 = args.model,
            w1 = args.w1,
            w2 = args.w2,
            w3 = args.w3
            )
 
    epoch_w = best_model_info_1['epoch']
    train_loss_unwrap_w = best_model_info_1['train_loss_unwrap']
    val_loss_unwrap_w = best_model_info_1['val_loss_unwrap']
    l1loss_train_unwrap = best_model_info_1['l1loss_train_unwrap']
    l2loss_train_unwrap = best_model_info_1['l2loss_train_unwrap']
    l1loss_val_unwrap = best_model_info_1['l1loss_val_unwrap']
    l2loss_val_unwrap = best_model_info_1['l2loss_val_unwrap']

    net_stat = best_model_info_1['state_dict']
    optimizer = best_model_info_1['optimizer_state_dict']
    dir_model = best_model_info_1['dir_model']
    result_all.append([args.w1, args.w2, args.w3,args.w4, epoch_w, train_loss_unwrap_w, l1loss_train_unwrap,l2loss_train_unwrap,l1loss_val_unwrap,\
                        l2loss_val_unwrap, val_loss_unwrap_w, net_stat, optimizer])
    print('#############train_loss_unwrap_w',train_loss_unwrap_w)
    print('##############Type of train_loss_unwrap_w:', type(train_loss_unwrap_w))


torch.cuda.empty_cache()   
    

