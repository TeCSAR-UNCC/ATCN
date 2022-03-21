'''
    Please refer to the following link more information about dataset:
    https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
'''

from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pathlib
__location__ = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(__location__))
from time_series_augmentation.utils.input_data import run_augmentation
from time_series_augmentation.utils.input_data import get_datasets, run_augmentation
import time_series_augmentation.utils.datasets as ds

def get_data(args):
    nb_class = ds.nb_classes(args.dataset)
    nb_dims = ds.nb_dims(args.dataset)
        
    # Load data
    x_train, y_train, x_test, y_test = get_datasets(args)
    
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)
        
    # Process data
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1])) 
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

    y_train = ds.class_offset(y_train, args.dataset)
    y_test = ds.class_offset(y_test, args.dataset)

    return x_train, y_train, x_test, y_test, nb_class

class UCR_DataSet(Dataset):
    r"""Torch Dataset implementation of UCR Time Series Classification Archive.
        Please refer to the following link more information about dataset:
        https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

        Args:
            x_train ([tensor]): [Time series input]
            y_train ([tensor]): [Classes output]
            train (bool, optional): [Is it for training or test?]. Defaults to True.
            transform ([type], optional): [A transform function if available. It is not supported yet.]. Defaults to None.
    """

    def __init__(self, x, y, nb_class, args, train=True, transform=None):
        self.train = train
        self.transform = transform

        self.num_class = nb_class
        
        if train:
            x, y, _ = run_augmentation(x, y, args)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

        self.length = self.x.shape[1]
        self.number_of_samples = self.x.shape[0]

    def __getitem__(self, idx):
        tmp_x = self.x[idx]
        tmp_y = self.y[idx]

        if self.transform is not None:
            tmp_x, tmp_y = self.transform(tmp_x, tmp_y)

        return tmp_x, tmp_y

    def __len__(self):
        return self.number_of_samples


if __name__ == "__main__":
    root_dir = '/mnt/AI_2TB/UCR/UCRArchive_2018/'
    dataset = 'Meat'

    args = {
        'dataset': dataset,
        'jitter': True,
        'augmentation_ratio': 4
    }

    dut = UCR_DataSet(root_dir=root_dir, dataset=dataset, args=args, train=True)
    X, Y = next(iter(dut))
    dataloader = DataLoader(dut, batch_size=4,
                            shuffle=True, num_workers=0)
    print("-> For dataset `{}` we have:\n\tLength: {:d}, Samples: {:d}, Class: {:d}".format(
        dataset, dut.length, dut.number_of_samples, dut.num_class))
