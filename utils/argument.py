from __future__ import print_function

import argparse


def argparser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--ucr2018', default=False,
                        action="store_true", help="Use UCR 2018")
    parser.add_argument('--ucr', default=False,
                        action="store_true", help="Use UCR 2018")
    parser.add_argument('--data-dir', type=str,
                        default="data", help="Data dir")
    parser.add_argument('-dt', '--dataset', metavar=str,
                        help='Name of dataset, e.g. ECG200')
    parser.add_argument('--preset-files', default=False,
                        action="store_true", help="Use preset files")

    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    '''
    parser.add_argument('--nhid', type=int, nargs='+', default=[320, 256, 256, 256, 256, 128, 128, 128, 128, 128, 64, 64, 64],
                        help='Number of hidden units per layer')
    parser.add_argument('--sdil', type=int, nargs='+', default=[1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8],
                        help='Size of dilation per layer')
    parser.add_argument('--skrn', type=int, nargs='+', default=[24, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8],
                        help='Size of kernel per layer')
    parser.add_argument('--input-scaling', type=int, nargs='+', default=[0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 1],
                        help='Size of kernel per layer')
    '''

    parser.add_argument('-mc', '--model-configuration', type=str, default='T2', choices=['TNH', 'T0NK', 'T0', 'TN1', 'T1', 'T2', 'TNHNK'],
                        help="ATCN model configuration from [TNH, T0, TN1, T1, T2, T0NK, TNHNK]")

    # parser.add_argument('--predict-size', type=int, default=3, dest='predict_size',
    #                    help='valid sequence length (default: 320)')

    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--conv2d', dest='conv2d', action='store_true')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
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
    parser.add_argument('--lr-decay', type=str, default='linear',
                        help='mode for learning rate decay')
    parser.add_argument('--step', type=int, default=10,
                        help='interval for learning rate decay in step mode')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--warmup', action='store_true',
                        help='set lower initial learning rate to warm up the training')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', required=True,
                        help='Path to save checkpoint')
    parser.add_argument('--cnv', dest='cnv', action='store_true',
                        help="Convert model from Conv2D to Conv1D")
    parser.add_argument('--clip', type=float, default=0.25,
                        help='Grdiant will be clipped to [-clip, clip] if --clip-grad argument is passed.')
    parser.add_argument('--clip-grad', dest='clip_grad', action='store_true',
                        help="Activate gradian clipping if it is passed.")
    parser.add_argument('--verbose', action='store_true',
                        default=False, help="verbose")

    # Augmentation
    parser.add_argument('--normalize_input', default=False,
                        action="store_true", help="Normalize between [-1,1]")
    parser.add_argument('--augmentation_ratio', type=int,
                        default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2,
                        help="Randomization seed")
    parser.add_argument('--jitter', default=False,
                        action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--j-sigma', type=float, default=0.02,
                        help='Sigma for jitter augmentation.')
    parser.add_argument('--scaling', default=False,
                        action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true",
                        help="Magnitude warp preset augmentation")
    parser.add_argument('--mw-sigma', type=float, default=0.1,
                        help='Sigma for magnitude_warp augmentation.')
    parser.add_argument('--timewarp', default=False,
                        action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False,
                        action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False,
                        action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False,
                        action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False,
                        action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False,
                        action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False,
                        action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true",
                        help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra-tag', type=str,
                        default="", help="Anything extra")
    parser.add_argument('--get-flops', default=False, action="store_true",
                        help="Will not start training. Only dumbs the flops and model sizes.")

    return parser.parse_args()
