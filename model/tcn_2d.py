"""
Copyright (c) 2020, University of North Carolina at Charlotte All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Authors: Reza Baharani - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
         Steven Furgurson - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
"""


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .axu import Mish, Swish
import math


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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, max_pooling=False):
        super(TemporalBlock, self).__init__()

        kernel_size = (1, kernel_size)
        dilation = (1, dilation)
        self.pad1 = nn.ConstantPad2d((padding, padding, 0, 0), 0)
        exp_ch = int(n_inputs)
        if n_inputs == n_outputs:
            self.do_res = True
            group_size = exp_ch
        else:
            self.do_res = False
            group_size = n_inputs

        self.max_pooling = max_pooling
        assert((self.do_res and max_pooling) == False)

        self.DSConv1 = nn.Sequential(
            nn.Conv2d(n_inputs, exp_ch, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(exp_ch),
            Swish(),
            nn.Conv2d(exp_ch, exp_ch, kernel_size=kernel_size,
                      stride=stride, padding=0, dilation=dilation, groups=group_size),
            nn.BatchNorm2d(exp_ch),
            Swish(),
            nn.Conv2d(exp_ch, n_outputs, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(n_outputs),
            Swish(),
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.act = Swish()

    def forward(self, x):

        res = x
        x = self.pad1(x)
        x = self.DSConv1(x)
        if self.max_pooling:
            x = self.pooling(x)
        if self.do_res:
            x = self.act(x + res)
        return x


class CNNMax(nn.Module):
    def __init__(self, ic, oc, k, d, p, dropout=0.2, active_downsampling=False):
        super(CNNMax, self).__init__()
        kernel_size = (1, k)
        dilation = (1, d)
        self.active_downsampling = active_downsampling
        self.pad1 = nn.ConstantPad2d((p, p, 0, 0), 0)
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=kernel_size, dilation=dilation,
                      bias=True, stride=1, padding=0),
            nn.BatchNorm2d(oc),
            Swish(),
        )
        self. downsampling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        #self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pad1(x)  # .type('torch.FloatTensor').cuda()
        x = self.block(x)
        if self.active_downsampling:
            x = self.downsampling(x)
        #x = self.dp(x)
        return x


class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, kernel_sizes, dilation_sizes, input_scalings, num_channels, input_ch=1, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        input_size = num_inputs
        for i, (kernel_size, dilation_size, input_scaling, in_channels) in enumerate(zip(kernel_sizes, dilation_sizes, input_scalings, num_channels)):
            in_channels = input_ch if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = getPaddingSize(input_size, 1, kernel_size, dilation_size)
            if input_scaling < 1:
                max_pooling = True
            else:
                max_pooling = False

            if i == 0:
                block = CNNMax(in_channels, out_channels,
                               kernel_size, dilation_size, padding, dropout, active_downsampling=False)
            else:
                block = TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                      padding=padding, dropout=dropout, max_pooling=max_pooling)
            layers += [block]
            input_size = math.ceil(input_size * input_scaling)

        self.network = nn.Sequential(*layers)
        self.network.apply(init_xavier)

    def forward(self, x):
        x = self.network(x)
        return x
