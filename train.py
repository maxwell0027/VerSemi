import os
import random
import time
import nibabel as nib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from utils1.losses import dice_loss, chaos_loss, ce_loss
from dataset.make_dataset import make_data_3d
from dataset.pancreas import Pancreas
from test_util import test_calculate_metric
from utils1 import statistic, ramps
from utils1.statistic import context_mask
from utils1.loss import DiceLoss, SoftIoULoss
from utils1.losses import FocalLoss
from utils1.ResampleLoss import ResampleLossMCIntegral
from vnet_odod_mix import VNet
from skimage.measure import label as label_cc
import logging
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--al_weight', type=float, default=0.1, help='the weight of aleatoric uncertainty loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--t', type=float, default=0.07, help='temperature')

args = parser.parse_args()

al_weight = args.al_weight

res_dir = 'checkpoint/'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

logging.basicConfig(filename=res_dir + "log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('New Exp :')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Parameters
num_class = 2
base_dim = 8

batch_size = 8   # 8
lr = 1e-3
beta1, beta2 = 0.5, 0.999


# log settings & test
pretraining_epochs = 1000
self_training_epochs = 1000
thres = 0.5
pretrain_save_step = 10
st_save_step = 10
pred_step = 10

r18 = False
dataset_name = 'unidataset'
data_root = '/data/userdisk1/qjzeng/semi_seg/UniSSM/preprocess/data'
cost_num = 3

alpha = 0.99
consistency = 0.1           
consistency_rampup = 40      



def sharpening(P):
    T = 1/args.t
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_consistency_weight(epoch, consistency_rampup=40):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def create_model(ema=False):
    net = nn.DataParallel(VNet(n_classes=2, n_branches=1))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_model_and_dataloader():
    """Net & optimizer"""
    net = create_model()
    ema_net = create_model(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

    """Loading Dataset"""
    logging.info("loading dataset")

    trainset_lab = Pancreas(data_root, dataset_name, split='train_lab', labeled=True)
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)      

    trainset_unlab = Pancreas(data_root, dataset_name, split='train_unlab', no_crop=True)
    unlab_loader = DataLoader(trainset_unlab, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pancreas(data_root, name='panc', split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    
    testset_la = Pancreas(data_root, name='la', split='test')
    test_loader_la = DataLoader(testset_la, batch_size=1, shuffle=False, num_workers=0)
    
    
    testset_sp = Pancreas(data_root, name='sp', split='test')
    test_loader_sp = DataLoader(testset_sp, batch_size=1, shuffle=False, num_workers=0)
    
    
    testset_lt = Pancreas(data_root, name='lt', split='test')
    test_loader_lt = DataLoader(testset_lt, batch_size=1, shuffle=False, num_workers=0)    

    return net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, test_loader_la, test_loader_sp, test_loader_lt


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    logging.info('Loaded from {}'.format(path))


def transform_label(label):
    s = label.shape
    res = torch.zeros(s[0], 2, s[1], s[2], s[3]).cuda()

    mask = (label == 0).long().unsqueeze(1).cuda()
    res[:, 0, :, :, :][mask] = 1

    mask  = (label == 1).long().unsqueeze(1).cuda()
    res[:, 1, :, :, :][mask] = 1

    return res


def pretrain(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, test_loader_la, test_loader_sp, test_loader_lt, start_epoch=1):
    save_path = Path(res_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)
    logging.info("Save path : {}".format(save_path))

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))

    maxdice1 = 0
    maxdice1_la = 0
    maxdice1_sp = 0
    maxdice1_lt = 0
    iter_num = 0
    DICE = DiceLoss(nclass=2)
    sub_bs = int(batch_size/2)
    
    #val_dice, maxdice1, max_flag = test(net, net, test_loader, maxdice1, task_id=1)
    #val_dice_la, maxdice1_la, max_flag_la = test(net, net, test_loader_la, maxdice1_la, task_id=2)
    #val_dice_sp, maxdice1_sp, max_flag_sp = test(net, net, test_loader_sp, maxdice1_sp, task_id=3)
    #val_dice_lt, maxdice1_lt, max_flag_lt = test(net, net, test_loader_lt, maxdice1_lt, task_id=4)    
    #exit()
    cnt = 0
    
    for epoch in tqdm(range(start_epoch, pretraining_epochs + 1), ncols=70):
        logging.info('\n')
        """Testing"""
        if epoch % pretrain_save_step == 0:
            # maxdice, _ = test(net, unlab_loader, maxdice, max_flag)
            val_dice, maxdice1, max_flag = test(net, net, test_loader, maxdice1, task_id=1)
            val_dice_la, maxdice1_la, max_flag_la = test(net, net, test_loader_la, maxdice1_la, task_id=2)
            val_dice_sp, maxdice1_sp, max_flag_sp = test(net, net, test_loader_sp, maxdice1_sp, task_id=3)
            val_dice_lt, maxdice1_lt, max_flag_lt = test(net, net, test_loader_lt, maxdice1_lt, task_id=4) 

            writer.add_scalar('pretrain/test_dice', val_dice, epoch)

            save_net_opt(net, optimizer, save_path / ('%d.pth' % epoch), epoch)
            logging.info('Save model : {}'.format(epoch))
            if max_flag:
                save_net_opt(net, optimizer, save_path / 'best.pth', epoch)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch)

        train_loss, train_dice= \
            AverageMeter(), AverageMeter()
        net.train()
        for step, (img, lab, task_id) in enumerate(lab_loader):
            img, lab = img.cuda(), lab.cuda()
            
            lab = lab==1
            lab = lab.long()
            
            output, _ = net(img, task_id)
            output_soft = F.softmax(output, dim=1)
            
            panc_idx = task_id == 1
            la_idx = task_id == 2
            spleen_idx = task_id == 3
            lungT_idx = task_id == 4
            
            
            panc_num = torch.sum(panc_idx)
            la_num = torch.sum(la_idx)
            spleen_num = torch.sum(spleen_idx)
            lungT_num = torch.sum(lungT_idx)
            
            '''
            # cutmix for LungT
            if lungT_num >= 2:
                img_lungT = img[lungT_idx]
                lab_lungT = lab[lungT_idx]
                lungT_reidx = random.sample(range(0,lungT_num), lungT_num)
                img_lungT_shuff = img_lungT[lungT_reidx]
                lab_lungT_shuff = lab_lungT[lungT_reidx]
                
                img_mask_lungT, lab_mask_lungT = context_mask(img_lungT, args.mask_ratio)
                
                img_lungT_mix = img_lungT * img_mask_lungT + img_lungT_shuff * (1 - img_mask_lungT)
                lab_lungT_mix = lab_lungT * lab_mask_lungT + lab_lungT_shuff * (1 - lab_mask_lungT)
                
                _, output_lungT = net(img_lungT_mix, 4*torch.ones(img_lungT_mix.shape[0]).view(-1))
                output_lungT_soft = F.softmax(output_lungT, dim=1)  
                loss_sup_lungT = dice_loss(output_lungT_soft[:,1,:,:,:], lab_lungT_mix == 1) + F.cross_entropy(output_lungT, lab_lungT_mix)
            '''
                             

            # chaos cutmix: pancreas / LA / spleen / LungT mixed together !chaos!
            img_a, img_b = img[:sub_bs], img[sub_bs:]
            lab_a, lab_b = lab[:sub_bs], lab[sub_bs:]            
            with torch.no_grad():
                #args.mask_ratio = random.uniform(0.3, 0.7)
                img_mask, lab_mask = context_mask(img_a, args.mask_ratio)
                

            img_mix = img_a * img_mask + img_b * (1 - img_mask)
            lab_mix = lab_a * lab_mask + lab_b * (1 - lab_mask)

            output_chaos, _ = net(img_mix, 5*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft = F.softmax(output_chaos, dim=1)
            
            # constrain between task-specific head and semantic-aware head
            task_id_batcha, task_id_batchb = task_id[:sub_bs], task_id[sub_bs:]
            
            lab_spec_b = lab_b * (1 - lab_mask)
            lab_spec_a = lab_a * lab_mask
            
            b_spec_output_chaos, _ = net(img_mix, task_id_batchb)
            b_spec_output_chaos_soft = F.softmax(b_spec_output_chaos, dim=1)
            a_spec_output_chaos, _ = net(img_mix, task_id_batcha)
            a_spec_output_chaos_soft = F.softmax(a_spec_output_chaos, dim=1)            

            # supervised loss
            loss_sup = dice_loss(output_soft[:,1,:,:,:], lab == 1) + F.cross_entropy(output, lab)
            loss_chaos = dice_loss(output_chaos_soft[:,1,:,:,:], lab_mix==1)
            
            # chech label for task-specific head
            for i_ in range(img_mix.shape[0]):
                if task_id_batcha[i_] == task_id_batchb[i_]:
                    lab_spec_a[i_] = lab_mix[i_]
                    lab_spec_b[i_] = lab_mix[i_]
            
            loss_spec_b = dice_loss(b_spec_output_chaos_soft[:,1,:,:,:], lab_spec_b==1)
            loss_spec_a = dice_loss(a_spec_output_chaos_soft[:,1,:,:,:], lab_spec_a==1)
            loss_spec = (loss_spec_a + loss_spec_b) / 2.
            
            
            '''
            # single task prediction when facing cutmixed data
            output_chaos_panc, _ = net(img_mix, 1*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_panc = F.softmax(output_chaos_panc, dim=1)    
            output_chaos_la, _ = net(img_mix, 2*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_la = F.softmax(output_chaos_la, dim=1) 
            output_chaos_sp, _ = net(img_mix, 3*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_sp = F.softmax(output_chaos_sp, dim=1)                          
            output_chaos_lt, _ = net(img_mix, 4*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_lt = F.softmax(output_chaos_lt, dim=1) 
            
            #print(lab_mix.shape)
            predictions = output_chaos_soft[:,1,:,:,:] > 0.5
            pred_panc = output_chaos_soft_panc[:,1,:,:,:] > 0.5
            pred_la = output_chaos_soft_la[:,1,:,:,:] > 0.5
            pred_sp = output_chaos_soft_sp[:,1,:,:,:] > 0.5
            pred_lt = output_chaos_soft_lt[:,1,:,:,:] > 0.5
            for i in range(lab_mix.shape[0]):
                image = img_mix[i,0,:,:,:].detach().cpu().data.numpy()
                label = lab_mix[i].detach().cpu().data.numpy()
                prediction = predictions[i].detach().cpu().data.numpy()
                predic_panc = pred_panc[i].detach().cpu().data.numpy()
                predic_la = pred_la[i].detach().cpu().data.numpy()
                predic_sp = pred_sp[i].detach().cpu().data.numpy()
                predic_lt = pred_lt[i].detach().cpu().data.numpy()
                nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_pred.nii.gz")
                nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_img.nii.gz")
                nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_gt.nii.gz")
                nib.save(nib.Nifti1Image(predic_panc.astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_panc.nii.gz")
                nib.save(nib.Nifti1Image(predic_la.astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_la.nii.gz")
                nib.save(nib.Nifti1Image(predic_sp.astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_sp.nii.gz")
                nib.save(nib.Nifti1Image(predic_lt.astype(np.float32), np.eye(4)), './CutMix_save_each/' + str(cnt) + "_lt.nii.gz")
                
                cnt += 1
            '''
            
            '''
            if lungT_num >= 2:
                loss = loss_sup + loss_chaos + loss_sup_lungT + loss_spec
            else:
                loss = loss_sup + loss_chaos + loss_spec
            '''
            
            
            
            
            
            '''
            
            # single task prediction when facing cutmixed data
            output_chaos_panc, _ = net(img_mix, 1*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_panc = F.softmax(output_chaos_panc, dim=1)    
            output_chaos_la, _ = net(img_mix, 2*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_la = F.softmax(output_chaos_la, dim=1) 
            output_chaos_sp, _ = net(img_mix, 3*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_sp = F.softmax(output_chaos_sp, dim=1)                          
            output_chaos_lt, _ = net(img_mix, 4*torch.ones(img_mix.shape[0]).view(-1))
            output_chaos_soft_lt = F.softmax(output_chaos_lt, dim=1) 
            
            
            ema_output_chaos_panc, _ = ema_net(img_mix, 1*torch.ones(img_mix.shape[0]).view(-1))
            ema_output_chaos_soft_panc = F.softmax(ema_output_chaos_panc, dim=1)    
            ema_output_chaos_la, _ = ema_net(img_mix, 2*torch.ones(img_mix.shape[0]).view(-1))
            ema_output_chaos_soft_la = F.softmax(ema_output_chaos_la, dim=1) 
            ema_output_chaos_sp, _ = ema_net(img_mix, 3*torch.ones(img_mix.shape[0]).view(-1))
            ema_output_chaos_soft_sp = F.softmax(ema_output_chaos_sp, dim=1)                          
            ema_output_chaos_lt, _ = ema_net(img_mix, 4*torch.ones(img_mix.shape[0]).view(-1))
            ema_output_chaos_soft_lt = F.softmax(ema_output_chaos_lt, dim=1) 
            ema_output_chaos_all, _ = ema_net(img_mix, 5*torch.ones(img_mix.shape[0]).view(-1))
            ema_output_chaos_soft_all = F.softmax(ema_output_chaos_all, dim=1) 

            
            #print(lab_mix.shape)
            predictions = output_chaos_soft[:,1,:,:,:] > 0.5
            pred_panc = output_chaos_soft_panc[:,1,:,:,:] > 0.5
            pred_la = output_chaos_soft_la[:,1,:,:,:] > 0.5
            pred_sp = output_chaos_soft_sp[:,1,:,:,:] > 0.5
            pred_lt = output_chaos_soft_lt[:,1,:,:,:] > 0.5
            
            ema_predictions = ema_output_chaos_soft_all[:,1,:,:,:] > 0.5
            ema_pred_panc = ema_output_chaos_soft_panc[:,1,:,:,:] > 0.5
            ema_pred_la = ema_output_chaos_soft_la[:,1,:,:,:] > 0.5
            ema_pred_sp = ema_output_chaos_soft_sp[:,1,:,:,:] > 0.5
            ema_pred_lt = ema_output_chaos_soft_lt[:,1,:,:,:] > 0.5
            
            for i in range(lab_mix.shape[0]):
                image = img_mix[i,0,:,:,:].detach().cpu().data.numpy()
                label = lab_mix[i].detach().cpu().data.numpy()
                prediction = predictions[i].detach().cpu().data.numpy()
                predic_panc = pred_panc[i].detach().cpu().data.numpy()
                predic_la = pred_la[i].detach().cpu().data.numpy()
                predic_sp = pred_sp[i].detach().cpu().data.numpy()
                predic_lt = pred_lt[i].detach().cpu().data.numpy()
                
                ema_prediction = ema_predictions[i].detach().cpu().data.numpy()
                ema_predic_panc = ema_pred_panc[i].detach().cpu().data.numpy()
                ema_predic_la = ema_pred_la[i].detach().cpu().data.numpy()
                ema_predic_sp = ema_pred_sp[i].detach().cpu().data.numpy()
                ema_predic_lt = ema_pred_lt[i].detach().cpu().data.numpy()
                
                nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "Uni_pred.nii.gz")
                nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "_img.nii.gz")
                nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "_gt.nii.gz")
                nib.save(nib.Nifti1Image(predic_panc.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "Uni_panc.nii.gz")
                nib.save(nib.Nifti1Image(predic_la.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "Uni_la.nii.gz")
                nib.save(nib.Nifti1Image(predic_sp.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "Uni_sp.nii.gz")
                nib.save(nib.Nifti1Image(predic_lt.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "Uni_lt.nii.gz")
                
                nib.save(nib.Nifti1Image(ema_prediction.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "dod_pred.nii.gz")
                nib.save(nib.Nifti1Image(ema_predic_panc.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "dod_panc.nii.gz")
                nib.save(nib.Nifti1Image(ema_predic_la.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "dod_la.nii.gz")
                nib.save(nib.Nifti1Image(ema_predic_sp.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "dod_sp.nii.gz")
                nib.save(nib.Nifti1Image(ema_predic_lt.astype(np.float32), np.eye(4)), './CutMix_Dod_UniSeMiv2/' + str(cnt) + "dod_lt.nii.gz")
                
                cnt += 1
            
            
            '''
            
            
            loss = loss_sup + loss_chaos + loss_spec
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            masks = get_mask(output)
            train_dice.update(statistic.uni_dice_ratio(masks, lab), 1)
            train_loss.update(loss.item(), 1)

            logging.info('epoch : %d, step : %d, loss_sup: %.4f, loss_chaos: %.4f, loss_spec: %.4f, train_dice: %.4f'\
                            % (epoch, step, loss_sup.item(), loss_chaos.item(), loss_spec.item(), train_dice.avg))

            writer.add_scalar('pretrain/loss_all', train_loss.avg, epoch * len(lab_loader) + step)
            writer.add_scalar('pretrain/train_dice', train_dice.avg, epoch * len(lab_loader) + step)
            update_ema_variables(net, ema_net, alpha, step)

        writer.flush()


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_mask(out, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label_cc(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda().long()


def train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader):
    save_path = Path(res_dir) / 'train'
    save_path.mkdir(exist_ok=True)
    #logging.info("Save path : ", save_path)

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))

    maxdice = 0
    maxdice1 = 0
    maxdice1_la = 0
    maxdice1_sp = 0
    maxdice1_lt = 0
    iter_num = 0
    DICE = DiceLoss(nclass=2)
    

    for epoch in tqdm(range(1, self_training_epochs+1)):
        logging.info('')
        writer.flush()

        if epoch % st_save_step == 0:
            net.eval()
            """Testing"""
            val_dice, maxdice1, max_flag = test(net, net, test_loader, maxdice1, task_id=1)
            val_dice_la, maxdice1_la, max_flag_la = test(net, net, test_loader_la, maxdice1_la, task_id=2)
            val_dice_sp, maxdice1_sp, max_flag_sp = test(net, net, test_loader_sp, maxdice1_sp, task_id=3)
            val_dice_lt, maxdice1_lt, max_flag_lt = test(net, net, test_loader_lt, maxdice1_lt, task_id=4) 

            """Save model"""
            if epoch > 0:
                save_net_opt(net, optimizer, str(save_path / ('{}.pth'.format(epoch))), epoch)
                save_net_opt(ema_net, optimizer, str(save_path / ('{}_ema.pth'.format(epoch))), epoch)
                logging.info('Save model : {}'.format(epoch))

            if max_flag:
                save_net_opt(net, optimizer, str(save_path / 'best.pth'), epoch)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch)

        net.train()
        ema_net.eval()

        for step, (data1, data2) in enumerate(zip(lab_loader, unlab_loader)):
            img_lb, lab, task_id_lb = data1
            img_lb, lab = img_lb.cuda(), lab.long().cuda()
            
            lab = lab==1
            lab = lab.long()
            
            img_un, _, task_id_un = data2
            img_un = img_un.cuda()
            bs_lb = img_lb.shape[0]
            bs_un = img_un.shape[0]

            # student prediction
            #img_input = torch.cat((img_lb, img_un), dim=0)
            #task_id = torch.cat((task_id_lb, task_id_un), dim=0)
            
            output_stu, _ = net(img_lb, task_id_lb)
            output_stu_soft = F.softmax(output_stu, dim=1)
            

            # supervised chaos cutmix: pancreas / LA / spleen / LungT mixed together !chaos!
            sub_bs = int(img_lb.shape[0]//2)
            img_a, img_b = img_lb[:sub_bs], img_lb[sub_bs:]
            lab_a, lab_b = lab[:sub_bs], lab[sub_bs:]            
            with torch.no_grad():
                args.mask_ratio = random.uniform(0.3, 0.6)
                sup_img_mask, sup_lab_mask = context_mask(img_a, args.mask_ratio)
            img_chaos_sup = img_a * sup_img_mask + img_b * (1 - sup_img_mask)
            lab_chaos_sup = lab_a * sup_lab_mask + lab_b * (1 - sup_lab_mask)

            sup_output_chaos, _ = net(img_chaos_sup, 5*torch.ones(img_chaos_sup.shape[0]).view(-1))
            sup_output_chaos_soft = F.softmax(sup_output_chaos, dim=1)

            
            # random mixed input, neither pure pancreas / la / sp / lt data
            with torch.no_grad():
                args.mask_ratio = random.uniform(0.3, 0.6)
                img_mask, lab_mask = context_mask(img_un, args.mask_ratio)
            img_chaos = img_un * img_mask + img_lb * (1 - img_mask)
            #img_mask, lab_mask = context_mask(img_un[:bs_un//2], args.mask_ratio)
            #img_chaos = img_un[:bs_un//2] * img_mask + img_un[bs_un//2:] * (1 - img_mask)

            output_chaos, _ = net(img_chaos, 5*torch.ones(img_chaos.shape[0]).view(-1))
            output_chaos_soft = F.softmax(output_chaos, dim=1)

            
            # chaos panc prediction
            output_chaos_panc, _ = net(img_chaos, 1*torch.ones(img_chaos.shape[0]).view(-1))
            output_chaos_panc_soft = F.softmax(output_chaos_panc, dim=1)   
            prob_panc_chaos = output_chaos_panc_soft[:,1,:,:,:]

            # chaos la prediction
            output_chaos_la, _ = net(img_chaos, 2*torch.ones(img_chaos.shape[0]).view(-1))
            output_chaos_la_soft = F.softmax(output_chaos_la, dim=1)   
            prob_la_chaos = output_chaos_la_soft[:,1,:,:,:]
            
            # chaos sp prediction
            output_chaos_sp, _ = net(img_chaos, 3*torch.ones(img_chaos.shape[0]).view(-1))
            output_chaos_sp_soft = F.softmax(output_chaos_sp, dim=1)   
            prob_sp_chaos = output_chaos_sp_soft[:,1,:,:,:]
            
            # chaos lt prediction
            output_chaos_lt, _ = net(img_chaos, 4*torch.ones(img_chaos.shape[0]).view(-1))
            output_chaos_lt_soft = F.softmax(output_chaos_lt, dim=1)   
            prob_lt_chaos = output_chaos_lt_soft[:,1,:,:,:]

            pred_chaos_panc_la = torch.maximum(prob_panc_chaos, prob_la_chaos)
            pred_chaos_sp_lt = torch.maximum(prob_sp_chaos, prob_lt_chaos)
            pred_chaos = torch.maximum(pred_chaos_panc_la, pred_chaos_sp_lt)
            pred_chaos_clip = torch.clip(pred_chaos, 0., 1.)
            
            #lab_chaos_comb = lab * lab_mask + pred_chaos_clip.detach() * (1 - lab_mask)
            
            consistency_weight_mix = consistency * get_current_consistency_weight(epoch)
            consistency_weight = get_current_consistency_weight(epoch)
                        
            # supervised loss
            loss_sup = dice_loss(output_stu_soft[:,1,:,:,:], lab == 1) + F.cross_entropy(output_stu, lab)
            loss_chaos_sup = dice_loss(sup_output_chaos_soft[:,1,:,:,:] , lab_chaos_sup==1)
            # unsupervised loss
            loss_chaos = dice_loss(output_chaos_soft[:,1,:,:,:] , pred_chaos_clip)
            loss_cons = dice_loss(output_chaos_soft[:,1,:,:,:] * (1 - img_mask) , (lab==1) * (1 - img_mask))
            

            loss = loss_sup + loss_chaos + loss_cons + loss_chaos_sup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha, iter_num + len(lab_loader) * pretraining_epochs)
                iter_num = iter_num + 1
                
            
            logging.info(
                'iteration %d : loss : %f, loss_sup: %f, loss_chaos: %f, loss_chaos_sup: %f, loss_cons: %f' %
                (iter_num, loss.item(), loss_sup.item(), loss_chaos.item(), loss_chaos_sup.item(), loss_cons.item() ))
              

        if epoch % st_save_step == 0:
            writer.add_scalar('val_dice', val_dice, epoch)


@torch.no_grad()
def test(net, ema_net, val_loader, maxdice=0, task_id=1):
    metrics = test_calculate_metric(net, ema_net, val_loader.dataset, task_id)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag


if __name__ == '__main__':
    set_random_seed(2333)
    net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, test_loader_la, test_loader_sp, test_loader_lt = get_model_and_dataloader()
    # First step: pretrain
    load_net_opt(ema_net, optimizer, './20Iter_final_results/pretrain/820.pth')
    load_net_opt(net, optimizer, '/data/userdisk1/qjzeng/semi_seg/UniSSMv5/constrain_20_bi/pretrain_/450.pth')
    pretrain(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, test_loader_la, test_loader_sp, test_loader_lt, start_epoch=1)
    
    
    # Second step: unlabeled data mining
    #load_net_opt(net, optimizer, './First_B_10Panc_La_Spleen_B10LungT/pretrain/570.pth')
    #load_net_opt(ema_net, optimizer, './First_B_10Panc_La_Spleen_B10LungT/pretrain/570.pth')
    #net.module.precls_conv.requires_grad = False
    #net.module.controller.requires_grad = False
    #train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader)
    

    logging.info(count_param(net))

    # best setting with one GPU:  pretrain (loss_sup + loss_chaos)    
    # weight: (loss_sup + consistency_weight*loss_un + consistency_weight_mix*loss_chaos)
