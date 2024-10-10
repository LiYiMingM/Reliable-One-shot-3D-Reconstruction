import os
from optparse import OptionParser
import torch
import time
import os.path
import scipy.io as sio
#from Network_HRNET import HighResolutionNet
#from Network_UNET import UNet
from dataset_read import get_dataloaders
# from UNET_RES import UNet ,WHOLE_NET
# from resnet_unet import WHOLE_NET
# from UNet import WHOLE_NET
from resnet_unet_DAnet import WHOLE_NET
import csv
' Definition of the needed parameters '
torch.cuda.empty_cache()
from train_func import AverageMeter


def get_args():

    parser.add_option('-e', '--result', dest='result', default=".../weights_single-1.0-1.0-100.0/test", help='folder of results')

    parser.add_option('-r', '--root', dest='root', default=".../statue_1605/", help='root directory')

    parser.add_option('-m', '--model', dest='model', default='.../weights_single-1.0-1.0-100.0.pth', help='folder for model/weights')

    parser.add_option('-i', '--input1', dest='input1', default='grating_4', help='folder of input')
   # parser.add_option('-j', '--input2', dest='input2', default='si_low_fre_grating', help='folder of input')
    parser.add_option('-g', '--ground_truth1', dest='gt_unwrap', default='unwrapped_phase', help='folder of ground truth')
    parser.add_option('-n', '--ground_truth2', dest='gt_phl3', default='phl_3', help='folder of ground truth')
    parser.add_option('-k', '--ground_truth3', dest='gt_wrap', default='wrapped_phase_4', help='folder of ground truth')


    (options, args) = parser.parse_args()
    return options
' Pass inputs through the Res-UNet '
def get_results(load_weights, dir_input1,dir_gt_phl3,  dir_gt_wrap, dir_gt_unwrap, resultdir):

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = WHOLE_NET(backbone='resnet34').to(device)

    # Load old weights
    checkpoint = torch.load(load_weights, map_location='cpu')

    
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key[7:]
        new_state_dict[new_key] = value
  
    # net.load_state_dict(checkpoint['state_dict'])
    net.load_state_dict(new_state_dict)
    # Load the dataset
    _,_,loader = get_dataloaders(dir_input1,dir_gt_phl3,  dir_gt_wrap, dir_gt_unwrap, batch_size=10,val_percent=0.1, test_size=0.1)
    # If resultdir does not exists make folder
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    net.eval()
    loss1_ave_phl3 = AverageMeter()
    loss2_ave_phl3 = AverageMeter()

    loss1_ave_wrap = AverageMeter()
    loss2_ave_wrap = AverageMeter()
    loss1_ave_unwrap = AverageMeter()
    loss2_ave_unwrap = AverageMeter()
    loss1_ave_total = AverageMeter()
    loss2_ave_total = AverageMeter()
    a1_ave = AverageMeter()
    a2_ave = AverageMeter()
    a3_ave = AverageMeter()
    with torch.no_grad():
        time1 = 0
        loss_l1 = torch.nn.L1Loss() 
        loss_l2 = torch.nn.MSELoss()
        path_csv = resultdir + "test_loss and others_all" + ".csv"
        path_csv_unwrap = resultdir + "test_loss_unwrap_only" + ".csv"
        header = ['l1_total_loss','rmse_total_loss', 'loss1_total_ave','loss2_total_ave','time cost now/second',
                  'l1_phl3_loss','rmse_phl3_loss', 'loss1_phl3_ave','loss2_phl3_ave',
                  'l1_wrap_loss','rmse_wrap_loss', 'loss1_wrap_ave','loss2_wrap_ave',    
                  'l1_unwrap_loss','rmse_unwrap_loss', 'loss1_unwrap_ave','loss2_unwrap_ave',
                 
                  ]
        header_unwrap = ['l1_unwrap_loss','rmse_unwrap_loss', 'loss1_unwrap_ave','loss2_unwrap_ave']        
        count=0
        for (input,gt_phl3,gt_wrap,gt_unwrap ) in loader:
             input,gt_phl3,gt_wrap,gt_unwrap = input.to(device), gt_phl3.to(device),gt_wrap.to(device),gt_unwrap.to(device)                                                 # Send data to GPU
             print("input",input.size())
             output_train_phl3,output_train_wrap,output_train_unwrap = net(input) # Forward
 

             for th in range(0, len(input)):
                time_start = time.time()

                loss1_phl3 = loss_l1(output_train_phl3 [th], gt_phl3[th])
                loss2_phl3 = torch.sqrt(loss_l2(output_train_phl3 [th], gt_phl3[th]))
                loss1_ave_phl3.update(loss1_phl3.item(),1) 
                loss2_ave_phl3.update(loss2_phl3.item(),1) 
       
                #wrapS
                loss1_wrap= loss_l1(output_train_wrap[th], gt_wrap[th])
                loss2_wrap= torch.sqrt(loss_l2(output_train_wrap[th], gt_wrap[th]))
                loss1_ave_wrap.update(loss1_wrap.item(),1) 
                loss2_ave_wrap.update(loss2_wrap.item(),1) 
                #loss_unwrap
                loss1_unwrap = loss_l1(output_train_unwrap [th], gt_unwrap[th])
                loss2_unwrap = torch.sqrt(loss_l2(output_train_unwrap [th], gt_unwrap[th]))
                loss1_ave_unwrap.update(loss1_unwrap.item(),1) 
                loss2_ave_unwrap.update(loss2_unwrap.item(),1) 
                #loss_total
                loss1_total= loss1_phl3 +loss1_wrap+loss1_unwrap
                loss2_total= loss2_phl3 +loss2_wrap+loss2_unwrap
                loss1_ave_total.update(loss1_total.item(),1) 
                loss2_ave_total.update(loss2_total.item(),1)     
                

                print(' loss1_ave_unwrap: ' + str(round(loss1_ave_unwrap.avg, 6)))   
                print(' loss2_ave_unwrap: ' + str(round(loss2_ave_unwrap.avg, 6)))  
                print(' loss1_ave_wrap: ' + str(round(loss1_ave_wrap.avg, 6)))   
                print(' loss2_ave_wrap: ' + str(round(loss2_ave_wrap.avg, 6))) 
                print(' loss1_ave_phl3: ' + str(round(loss1_ave_phl3.avg, 6)))   
                print(' loss2_ave_phl3: ' + str(round(loss2_ave_phl3.avg, 6))) 

                time_cost_now = time.time() - time_start
                values = [loss1_total.item(), loss2_total.item(),round(loss1_ave_total.avg, 6),round(loss2_ave_total.avg, 6),time_cost_now,
                          loss1_phl3.item(), loss2_phl3.item(),round(loss1_ave_phl3.avg, 6),round(loss2_ave_phl3.avg, 6),                   
                          loss1_wrap.item(), loss2_wrap.item(),round(loss1_ave_wrap.avg, 6),round(loss2_ave_wrap.avg, 6),                          
                          loss1_unwrap.item(), loss2_unwrap.item(),round(loss1_ave_unwrap.avg, 6),round(loss2_ave_unwrap.avg, 6)
                              
                          ]

                values_unwrap = [                      
                          loss1_unwrap.item(), loss2_unwrap.item(),round(loss1_ave_unwrap.avg, 6),round(loss2_ave_unwrap.avg, 6)
                                                     
                          ]

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


                if os.path.isfile(path_csv_unwrap) == False:
                    file = open(path_csv_unwrap, 'w', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(header_unwrap)
                    writer_csv.writerow(values_unwrap)
                else:
                    file = open(path_csv_unwrap, 'a', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(values_unwrap)
                file.close()




                input1 = input[th]#[1, 256, 256]
                input1_1 = torch.unsqueeze(input1, 0)
        
                output_train_phl3_1,output_train_wrap_1,output_train_unwrap_1= net(input1_1)

                input1_numpy = input1_1.squeeze(0).squeeze(0).cpu().numpy()
                #phl3
                output_numpy_phl3 = output_train_phl3_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_phl3 = gt_phl3[th].squeeze(0).cpu().numpy()
            
                #wrap
                output_numpy_wrap = output_train_wrap_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_wrap = gt_wrap[th].squeeze(0).cpu().numpy()
                #unwrap
                output_numpy_unwrap = output_train_unwrap_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_unwrap = gt_unwrap[th].squeeze(0).cpu().numpy()


                filename = resultdir + (str(count + 1).zfill(6)) + '-results.mat'
                sio.savemat(filename, {'input': input1_numpy,'output_phl3': output_numpy_phl3, 'gt_phl3': gt_numpy_phl3,
                                       'output_wrap': output_numpy_wrap, 'gt_wrap': gt_numpy_wrap,
                                       'output1': output_numpy_unwrap, 'gt1': gt_numpy_unwrap
                                         })
            
                count +=1
                time_end = time.time()
                time1 = time1 + (time_end - time_start)
        print('totally cost', time1)



' Run the application '
if __name__ == '__main__':
    args = get_args()
    get_results(load_weights = args.root + args.model,

                dir_input1=args.root + args.input1 + '/',
                dir_gt_unwrap = args.root +args.gt_unwrap+'/',
                dir_gt_phl3 = args.root +args.gt_phl3+'/',
                dir_gt_wrap = args.root +args.gt_wrap+'/',
                resultdir=args.root + args.result + "/")
    
