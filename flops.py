import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from utils import argparser
from model.net import ATCN
from collections import OrderedDict
import os
from data_processing.utils import data_generator

def main(args):
    with torch.cuda.device(4):
        input_size = 784
        print('=> Making network ready for input size: {}'.format(input_size))
        #train_loader, val_loader = data_generator(args.data, args.batch_size)

        # create model

        channel_sizes = args.nhid  # [args.nhid[0]] * (args.levels-1)
        kernel_size = args.skrn
        dropout = args.dropout


        print("=> creating model")
        # Load model for training and inference
        model = ATCN(input_size, args.predict_size, kernel_size,
                    args.sdil, args.input_scaling, channel_sizes,
                    inp_ch=1, dropout=dropout, conv2d=args.conv2d)

        # Make sure we're loading the same network.
        best_model = os.path.join(args.checkpoint, "model_best.pth.tar")
        source_state = torch.load(best_model)
        model.load_state_dict(source_state['state_dict'])

        model.cuda()
        macs, params = get_model_complexity_info(model, (1, 1, input_size), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == "__main__":
    args = argparser()
    main(args)
