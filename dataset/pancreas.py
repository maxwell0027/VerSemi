import os
from pathlib import Path
import random

import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

'''
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
'''



def get_dataset_path():
    files = ['train_lab_20.txt', 'train_unlab_20.txt', 'test.txt', 'test_la.txt', 'test_spleen.txt', 'test_lung.txt',\
             'panc_lab.txt', 'la_lab.txt', 'sp_lab.txt', 'lt_lab.txt']
    return ['./datalist/'+ f for f in files]


class Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, name, split, no_crop=False, TTA=False, require_mask=False, labeled=False):
        self._base_dir = base_dir
        self.split = split
        self.require_mask = require_mask
        self.image_name = None
        self.labeled = labeled
        self.name = name
        
        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            ToTensor()
        ])
        
        
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path()

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'panc_lab':
            data_path = data_list_paths[6]
            self.transform = tr_transform
        elif split == 'la_lab':
            data_path = data_list_paths[7]
            self.transform = tr_transform
        elif split == 'sp_lab':
            data_path = data_list_paths[8]
            self.transform = tr_transform
        elif split == 'lt_lab':
            data_path = data_list_paths[9]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform
        else:
            data_path = data_list_paths[2]
            data_path_la = data_list_paths[3]
            data_path_sp = data_list_paths[4]
            data_path_lt = data_list_paths[5]
            self.transform = test_transform
    
            if self.name == 'panc':
                data_path = data_path
            elif self.name == 'la':
                data_path = data_path_la
            elif self.name == 'sp':
                data_path = data_path_sp
            elif self.name == 'lt':
                data_path = data_path_lt
        
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_name = [item.strip() for item in self.image_list]
        self.image_list = [self._base_dir + "/{}".format(item.strip()) + '.h5' for item in self.image_list]

        #if self.labeled == True:
        #    self.image_list = self.image_list + self.image_list

        #random.shuffle(self.image_list)
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab':
            return len(self.image_list) * 5
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        image_name = image_path.split('/')[-1]
            
        if image_name[:4] == 'data':          # pancreas
            task_id = 1
        elif image_name[:6] == 'spleen':
            task_id = 3
        elif image_name[:4] == 'lung':
            task_id = 4
        else:                                 # LA
            task_id = 2
            
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        if self.require_mask:
            mask = (label > 0).astype(np.uint8)
            samples = image, label, mask
            if self.transform:
                tr_samples = self.transform(samples)
            image_, label_, mask_ = tr_samples
            return image_.float(), label_.long(), mask_.long(), task_id
         
        else:
            samples = image, label
            if self.transform:
                tr_samples = self.transform(samples)
            image_, label_ = tr_samples
            return image_.float(), label_.long(), task_id
        


class MaxCenterCrop(object):
    def __init__(self, scale=16):
        self.output_scale = scale

    def _get_transform(self, label):
        max_v = max(label.shape)
        n = (max_v // self.output_scale)
        output_size = n * self.output_scale

        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= output_size[0] or x.shape[1] <= output_size[1] or x.shape[2] <= output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def _get_transform(self, x):
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        def do_transform(image):
            image = np.rot90(image, k)
            image = np.flip(image, axis=axis).copy()
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def _get_transform(self, x):
        noise = np.clip(self.sigma * np.random.randn(x.shape[0], x.shape[1], x.shape[2]), -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        def do_transform(image):
            image = image + noise
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) if i == 0 else s for i, s in enumerate(samples)]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


if __name__ == '__main__':
    pass
