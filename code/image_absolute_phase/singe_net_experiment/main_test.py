import os
from optparse import OptionParser
import torch
import time
import os.path
import scipy.io as sio
t
from dataset_read import get_dataloaders


from resnet_unet import Resnet_Unet
import csv
' Definition of the needed parameters '
torch.cuda.empty_cache()
from train_func import AverageMeter


def get_args():
    parser = OptionParser()

    parser.add_option('-e', '--result', dest='result', default=".../test", help='folder of results')
    parser.add_option('-r', '--root', dest='root', default=".../statue_1605/", help='root directory')

    parser.add_option('-m', '--model', dest='model', default='.../weights_single.pth', help='folder for model/weights')

    parser.add_option('-i', '--input1', dest='input1', default='grating_4', help='folder of input')

    parser.add_option('-g', '--ground truth1', dest='gt1', default='unwrapped_phase', help='folder of ground truth')

    (options, args) = parser.parse_args()
    return options

' Pass inputs through the Res-UNet '
def get_results(load_weights, dir_input1,dir_gt1,resultdir,batch_size=2,val_percent=0.1 ):


    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:1" if use_cuda else "cpu")


    net = Resnet_Unet(backbone='resnet50').to(device)

  # # using multiple GPUs
    checkpoint = torch.load(load_weights, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key[7:]
        new_state_dict[new_key] = value  
    net.load_state_dict(new_state_dict)    

    # # using single GPU
    # net.load_state_dict(checkpoint['state_dict']) 


    # Load the dataset
    _,_,loader = get_dataloaders(dir_input1, dir_gt1,batch_size=4,val_percent=0.1)
    # If resultdir does not exists make folder
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    net.eval()
    loss1_ave = AverageMeter()
    loss2_ave = AverageMeter()
    a1_ave = AverageMeter()
    a2_ave = AverageMeter()
    a3_ave = AverageMeter()
    with torch.no_grad():
        time1 = 0
        loss_l1 = torch.nn.L1Loss() 
        loss_l2 = torch.nn.MSELoss()
        path_csv = resultdir +"test_loss and others" + ".csv"
        header = ['l1 loss','rmse loss', 'loss1_ave','loss2_ave','time cost now/second','a1','a2','a3','a1_ave','a2_ave','a3_ave']
        count=0
        for (input, gt1) in loader:
            input, gt1= input.to(device), gt1.to(device)#gt1 torch.Size([600, 1, 256, 256])

            output11 = net(input)
            print("len(input)",len(input))
            for th in range(0, len(input)):
                print("th",th)
                time_start = time.time()

                loss1 = loss_l1(output11[th], gt1[th])
                loss2 = torch.sqrt(loss_l2(output11[th], gt1[th]))                
                loss1_ave.update(loss1.item(),1) 
                loss2_ave.update(loss2.item(),1) 
                print(' loss1_ave: ' + str(round(loss1_ave.avg, 6)))   
                print(' loss2_ave: ' + str(round(loss2_ave.avg, 6)))  
 
                thresh = torch.maximum((gt1[th]/output11[th]), (output11[th] / gt1[th]))
                a1 = (thresh < 1.25).to(torch.float32).mean()
                a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
                a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
                a1_ave.update(a1.item(),1) 
                a2_ave.update(a2.item(),1) 
                a3_ave.update(a3.item(),1) 
                print(' a1_ave: ' + str(round(a1_ave.avg, 4)))   
                print(' a2_ave: ' + str(round(a2_ave.avg, 4)))  
                print(' a3_ave: ' + str(round(a3_ave.avg, 4)))  


            
                time_cost_now = time.time() - time_start
                values = [loss1.item(), loss2.item(),round(loss1_ave.avg, 6),round(loss2_ave.avg, 6),time_cost_now,
                          a1.item(),a2.item(),a3.item(),round(a1_ave.avg, 4),round(a2_ave.avg, 4),round(a3_ave.avg, 4)]
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

                input1 = input[th]#[1, 256, 256]
                input1 = torch.unsqueeze(input1, 0)
                output1= net(input1)[0]
                input1_numpy = input1.squeeze(0).squeeze(0).cpu().numpy()
                output_numpy1 = output1.squeeze(0).cpu().numpy()
                gt_numpy1 = gt1[th].squeeze(0).cpu().numpy()
                filename = resultdir + (str(count + 1).zfill(6)) + '-results.mat'
                sio.savemat(filename, {'input1': input1_numpy,'output1': output_numpy1, 'gt1': gt_numpy1
                                         })

                count +=1
                time_end = time.time()
                time1 = time1 + (time_end - time_start)
        print('totally cost', time1)



' Run the application '
if __name__ == '__main__':
    args = get_args()
    get_results(load_weights = args.root + args.model,
                dir_input1 = args.root +args.input1+"/", 
                dir_gt1 = args.root +args.gt1+'/',
                resultdir=args.root + args.result + "/")
    
