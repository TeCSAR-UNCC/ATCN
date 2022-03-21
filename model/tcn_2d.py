import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
from .axu import Mish, Swish


def getPaddingSize(oi, s, k, d):
    #o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    p = math.ceil(0.5*((oi-1)*s-oi+k+((k-1)*(d-1))))
    return p


def init_xavier(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, max_pooling=False, conv2d=True):
        super(TemporalBlock, self).__init__()

        exp_ch = int(2*n_inputs)
        if n_inputs == n_outputs:
            self.do_res = True
            group_size = exp_ch
        else:
            self.do_res = False
            group_size = n_inputs

        self.max_pooling = max_pooling
        assert((self.do_res and max_pooling) == False)

        if conv2d:
            kernel_size = (1, kernel_size)
            dilation = (1, dilation)
            self.pad1 = nn.ConstantPad2d((padding, padding, 0, 0), 0)
            self.DSConv1 = nn.Sequential(
                nn.Conv2d(n_inputs, exp_ch, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm2d(exp_ch),
                Swish(),
                nn.Dropout(p=dropout),
                nn.Conv2d(exp_ch, exp_ch, kernel_size=kernel_size,
                          stride=stride, padding=0, dilation=dilation, groups=group_size),
                nn.BatchNorm2d(exp_ch),
                Swish(),
                nn.Dropout(p=dropout),
                nn.Conv2d(exp_ch, n_outputs, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm2d(n_outputs),
                Swish(),
            )
            #self.pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        else:
            kernel_size = kernel_size
            dilation = dilation
            self.pad1 = nn.ConstantPad1d((padding, padding), 0)
            self.DSConv1 = nn.Sequential(
                nn.Conv1d(n_inputs, exp_ch, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm1d(exp_ch),
                Swish(),
                nn.Dropout(p=dropout),
                nn.Conv1d(exp_ch, exp_ch, kernel_size=kernel_size,
                          stride=stride, padding=0, dilation=dilation, groups=group_size),
                nn.BatchNorm1d(exp_ch),
                Swish(),
                nn.Dropout(p=dropout),
                nn.Conv1d(exp_ch, n_outputs, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm1d(n_outputs),
                Swish(),
            )
            #self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.do = nn.Dropout(p=dropout)
        self.act = Swish()

    def forward(self, x):

        res = x  # if self.downsample is None else self.downsample(x)
        x = self.pad1(x)
        x = self.DSConv1(x)
        if self.max_pooling:
            assert False
            #x = self.pooling(x)
        if self.do_res:
            x = self.act(x + res)
        x = self.do(x)
        #x = self.res_path(x, res)
        return x


class CNNMax(nn.Module):
    def __init__(self, ic, oc, k, d, p, dropout=0.2, conv2d=True):
        super(CNNMax, self).__init__()
        if conv2d:
            kernel_size = (1, k)
            dilation = (1, d)
            self.pad1 = nn.ConstantPad2d((p, p, 0, 0), 0)
            self.block = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=kernel_size, dilation=dilation,
                          bias=True, stride=1, padding=0),
                nn.BatchNorm2d(oc),
                Swish(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=2),
                nn.Dropout(dropout)
            )
        else:
            kernel_size = k
            dilation = d
            self.pad1 = nn.ConstantPad1d((p, p), 0)
            self.block = nn.Sequential(
                nn.Conv1d(ic, oc, kernel_size=kernel_size, dilation=dilation,
                          bias=True, stride=1, padding=0),
                nn.BatchNorm1d(oc),
                Swish(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = self.pad1(x)  # .type('torch.FloatTensor').cuda()
        x = self.block(x)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, kernel_sizes, dilation_sizes, input_scaling, num_channels, dropout=0.2, conv2d=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        input_size = num_inputs
        for i, (kernel_size, dilation_size, input_scaling, in_channels) in enumerate(zip(kernel_sizes, dilation_sizes, input_scaling, num_channels)):
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = getPaddingSize(input_size, 1, kernel_size, dilation_size)
            if input_scaling < 1:
                max_pooling = True
            else:
                max_pooling = False

            if i == 0:
                block = CNNMax(in_channels, out_channels,
                               kernel_size, dilation_size, padding, dropout, conv2d=conv2d)
            else:
                block = TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                      padding=padding, dropout=dropout, max_pooling=max_pooling, conv2d=conv2d)
            layers += [block]
            input_size = input_size * input_scaling

        self.network = nn.Sequential(*layers)
        self.network.apply(init_xavier)

    def forward(self, x):
        x = self.network(x)
        return x    
