from torch.utils import data

import pandas as pd

import torch as t

import os
import sys
import numpy as np

from .data_utils import *


class VOXDataset(data.Dataset):
    def __init__(self, root, train=True, crop_size=32, move=3):
        '''The classification-only dataset.
        :param crop_size: the input size
        :param move: the random move
        '''
        if train:
            self.transform = Transform(crop_size, move)

            df = pd.read_csv(os.path.join(root, 'train.csv'))
            self.mean=np.load(os.path.join(root, 'train_mean.npy'))
            self.std=np.load(os.path.join(root, 'train_std.npy'))
            self.label = df['labels'].values
            self.names = df['names'].values
        else:

            df = pd.read_csv(os.path.join(root, 'val.csv'))
            self.mean=np.load(os.path.join(root, 'val_mean.npy'))
            self.std=np.load(os.path.join(root, 'val_std.npy'))
            self.label = df['labels'].values
            self.names = df['names'].values
            self.transform = Transform(crop_size, move=move)


        self.voxels = [os.path.join(root, 'train_val/%s.npz' % name) for name in self.names]

    def __getitem__(self, item):
        name = self.names[item]

        with np.load(self.voxels[item]) as npz:
            voxel = npz['voxel'].copy()
            voxel=(voxel-self.mean)/self.std
            voxel = self.transform(voxel)
            voxel = t.from_numpy(voxel.copy())
            voxel = voxel.float()
        label = self.label[item].copy()

        label = t.tensor(label)

        return voxel, label

    def __len__(self):
        return len(self.names)


class TestDataset(data.Dataset):

    def __init__(self, root, crop_size=32,move=0):
        df = pd.read_csv(os.path.join(root, 'test.csv'))
        self.mean = np.load(os.path.join(root, 'test_mean.npy'))
        self.std = np.load(os.path.join(root, 'test_std.npy'))
        self.names = df['name'].values
        print(root)
        self.voxels = [os.path.join(root, "test", '%s.npz' % name) for name in self.names]
        self.transform = Transform(crop_size, move=move)

    def __getitem__(self, item):
        with np.load(self.voxels[item]) as npz:
            voxel = npz['voxel'].copy()
            voxel=(voxel-self.mean)/self.std
            voxel = self.transform(voxel)
            voxel = t.from_numpy(voxel.copy())
            voxel = voxel.float()
        return voxel

    def __len__(self):
        return len(self.names)


class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotate 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = (size, size, size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_shift(shape, self.move)

            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)

            #arr_ret = rotate(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1

            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=0)

            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotate(aux_ret, angle=angle)
                #aux_ret = rotate(aux_ret, angle=angle)

                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=0)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
