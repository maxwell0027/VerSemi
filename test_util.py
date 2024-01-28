import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataset.pancreas import Pancreas
import os
import logging

def test_all_case(net, ema_net, image_list, task_id, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None,
                  preproc_fn=None):
    total_metric = 0.0
    cnt = 0
    for image_path in tqdm(image_list):

        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        label = label==1

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, ema_net, image, task_id, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        # print(single_metric)
        total_metric += np.asarray(single_metric)
        # print(str(cnt) + ", {}, {}, {}. {}".format(single_metric[0], single_metric[1], single_metric[2], single_metric[3]))


        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_gt.nii.gz")
        cnt += 1
    avg_metric = total_metric / len(image_list)

    logging.info('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, ema_net, image, task_id, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                out, _ = net(test_patch, torch.tensor([int(task_id)]))
                out_, _ = ema_net(test_patch, torch.tensor([int(task_id)]))
                
                y1 = (out+out_)/2.
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]

                '''
                y1 = net(test_patch)[0]
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :2, :, :, :]    # for pancreas; 
                #y = np.concatenate((np.expand_dims(y[0, 0, :, :, :], axis=0),np.expand_dims(y[0, 2, :, :, :], axis=0)),axis=0)     # for la
                '''
                
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_calculate_metric(net, ema_net, test_dataset, task_id, num_classes=2, save_result=False, test_save_path='./CutMix_Dod_UniSeMiv2'):
    net.eval()
    image_list = test_dataset.image_list

    if save_result:
        test_save_path = Path(test_save_path)
        test_save_path.mkdir(exist_ok=True)

    avg_metric = test_all_case(net, ema_net, image_list, task_id, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=4,
                               save_result=save_result, test_save_path=str(test_save_path) + '/')
    return avg_metric


def test_calculate_metric_LA(net, ema_net, test_dataset, num_classes=2, save_result=False, test_save_path='./save'):
    net.eval()
    # with open("/home/xiangjinyi/semi_supervised/alnet/data_lists_cora/LA_dataset/test_whole.list", 'r') as f:
    #     image_list = f.readlines()
    # image_list = [item.replace('\n', '') for item in image_list]
    # image_list = [os.path.join("../LA_dataset", item, "mri_norm2.h5") for item in image_list]

    image_list = test_dataset.image_list
    avg_metric = test_all_case(net, ema_net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=18, stride_z=4,
                               save_result=save_result, test_save_path=test_save_path)    # (112, 112, 80)
    return avg_metric


if __name__ == '__main__':
    import os
    from train_panc import get_model_and_dataloader, load_net_opt, test

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    num_classes = 2
    res_dir = './result/pancreas_3d_VNet/st_model'

    split_name = 'pancreas'
    data_root = '/data/DataSets/pancreas_pad25'
    net, ema_net, optimizer, lab_loader, unlab_loader, test_loader = get_model_and_dataloader()
    dataset = Pancreas(data_root, split_name, split='test')
    load_net_opt(net, optimizer, res_dir + '/best.pth')
    metric = test_calculate_metric(net, dataset)
    print(metric)
