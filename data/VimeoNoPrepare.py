'''
Vimeo7 dataset
support reading images from lmdb and image folder 
'''
import os
import random
import pickle
import logging
import numpy as np
import lmdb
import cv2
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.generate_LR import downsample
from datasets.create_lmdb import get_keys

logger = logging.getLogger('base')


class VimeoDataset(Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    '''

    def __init__(self, config):
        super(VimeoDataset, self).__init__()
        self.config = config
        self.scale = self.config['scale']
        self.num_frames = self.config['num_frames']
        self.HR_crop_size = self.config['crop_size']
        self.LR_crop_size = self.HR_crop_size // self.scale
        self.HR_image_shape = tuple(self.config['image_shape'])
        self.LR_image_shape = (3, self.config['image_shape'][1] // self.scale, self.config['image_shape'][2] // self.scale)

        # temporal augmentation
        self.random_reverse = config['random_reverse']
        logger.info('Temporal augmentation with random reverse is {}.'.format(self.random_reverse))

        self.LR_num_frames = 1 + self.num_frames // 2
        assert self.LR_num_frames > 1, 'Error: Not enough LR frames to interpolate'

        self.LR_index_list = [i * 2 for i in range(self.LR_num_frames)]

        self.HR_root = config['dataroot_HR']

        # Load image keys
        self.HR_paths = list(get_keys(config['list']))
     
        assert self.HR_paths, 'Error: HR path is empty.'

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_env = lmdb.open(self.config['dataroot_HR'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LR_env = lmdb.open(self.config['dataroot_LR'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def read_img(self, path, size=None):
        """Read image using cv2 or from lmdb.

        Args:
            env: lmdb env. If None, read using cv2.
            path (str): key of lmdb file or path to the image.
            size (tuple, optional): Image size (C, H, W). Defaults to None.

        Returns:
            array: (H, W, C) BGR image. 
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def augment(self, img_list, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in img_list]

    def __getitem__(self, index):

        key = self.HR_paths[index]
        name_a, name_b = key.split('_')

        # Get frame list
        HR_frames_list = list(range(1, self.num_frames + 1))
        if self.random_reverse and random.random() < 0.5:
            HR_frames_list.reverse()
        LR_frames_list = [HR_frames_list[i] for i in self.LR_index_list]

        # Get HR images
        img_HR_dict = {}
        img_HR_list = []
        for v in HR_frames_list:
            img_HR = self.read_img(os.path.join(self.HR_root, name_a, name_b, 'im{}.png'.format(v)))
            img_HR_list.append(img_HR)
            img_HR_dict[v] = img_HR
                
        # Get LR images
        img_LR_list = []
        for v in LR_frames_list:
            img_LR = downsample(img_HR_dict[v], self.scale)
            img_LR_list.append(img_LR)

        _, H, W = self.LR_image_shape
        # Randomly crop
        rnd_h = random.randint(0, max(0, H - self.LR_crop_size))
        rnd_w = random.randint(0, max(0, W - self.LR_crop_size))
        img_LR_list = [v[rnd_h:rnd_h + self.LR_crop_size, rnd_w:rnd_w + self.LR_crop_size, :] for v in img_LR_list]
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
        img_HR_list = [v[rnd_h_HR:rnd_h_HR + self.HR_crop_size, rnd_w_HR:rnd_w_HR + self.HR_crop_size, :] for v in img_HR_list]

        # Augmentation - flip, rotate
        img_list = img_LR_list + img_HR_list
        img_list = self.augment(img_list, self.config['use_flip'], self.config['use_rot'])
        img_LR_list = img_list[0:-self.num_frames]
        img_HR_list = img_list[-self.num_frames:]

        # Stack LR images to NHWC, N is the frame number
        img_LRs = np.stack(img_LR_list, axis=0)
        img_HRs = np.stack(img_HR_list, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HRs = img_HRs[:, :, :, [2, 1, 0]]
        img_LRs = img_LRs[:, :, :, [2, 1, 0]]
        img_HRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HRs, (0, 3, 1, 2)))).float()
        img_LRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LRs, (0, 3, 1, 2)))).float()
        return {'LRs': img_LRs, 'HRs': img_HRs, 'key': key}

    def __len__(self):
        return len(self.HR_paths)
