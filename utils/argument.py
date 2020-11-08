from __future__ import print_function

import argparse


def argparser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-d', '--data', metavar='PATH', required=True,
                        help='Path to h5 file dataset')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--nhid', type=int, nargs='+', default=[8,  16, 16, 24, 24, 32, 32],
                        help='Number of hidden units per layer')
    parser.add_argument('--sdil', type=int, nargs='+', default=[1,  2,  2,  4,  4,  6,  6],
                        help='Size of dilation per layer')
    parser.add_argument('--skrn', type=int, nargs='+', default=[25, 13, 13,  7, 7,   5, 5],
                        help='Size of kernel per layer')
    parser.add_argument('--input-scaling', type=int, nargs='+', default=[0.5, 1, 1, 0.5, 1,  0.5, 1],
                        help='Size of kernel per layer')
    parser.add_argument('--predict-size', type=int, default=10, dest='predict_size',
                        help='valid sequence length (default: 320)')
    parser.add_argument('--ksize', type=int, default=8,
                        help='kernel size (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--conv2d', dest='conv2d', action='store_true')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--lr-decay', type=str, default='linear',
                        help='mode for learning rate decay')
    parser.add_argument('--step', type=int, default=10,
                        help='interval for learning rate decay in step mode')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--warmup', action='store_true',
                        help='set lower initial learning rate to warm up the training')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', required=True,
                        help='Path to save checkpoint')

    return parser.parse_args()
