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

import torch.nn.functional as F
from torch import nn
from .tcn_2d import TemporalConvNet2D


class ATCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes, dilation_sizes, input_scalings, num_channels, inp_ch=1, dropout=0.3, conv2d=False):
        super(ATCN, self).__init__()
        self.conv2d = conv2d
        if self.conv2d:
            self.tcn = TemporalConvNet2D(
                input_size, kernel_sizes, dilation_sizes, input_scalings, num_channels, input_ch=inp_ch, dropout=dropout)
        self.downsampler = nn.AdaptiveMaxPool2d((1, 1))
        self.dp = nn.Dropout(dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(inputs)
        if self.conv2d:
            x = self.downsampler(x)
        x = x.view((x.size(0), -1))
        x = self.dp(x)
        x = self.linear(x)

        return x