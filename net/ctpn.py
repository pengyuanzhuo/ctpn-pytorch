# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.vgg import vgg16_bn


class Im2col(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        '''
        conv2d without learnable parameters
        '''
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        h = x.shape[2] # x.shape = (N, C, H, W)
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride) # x.shape=(N, C*kernel_size, L)
        x = x.reshape((x.shape[0], x.shape[1], h, -1)) # channel=C*kernel_size, shape=(N, FN, h, w)
        return x


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, bidirectional=True):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute((0, 2, 3, 1)) # (N, C, H, W) => (N, H, W, C)
        N, H, W, C = x.size()
        x = x.reshape((N*H, W, C)) # (N, H, W, C) => (NH, W, C)
        x, _ = self.lstm(x) # shape=(batch, seq_len, num_directions*hidden_size)=(NH, W, 256)
        x = x.reshape((N, H, W, x.shape[-1])) # (NH, W, 256) => (N, H, W, 256)
        x = x.permute((0, 3, 1, 2))
        # assert x.shape[1] == 256

        return x


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.feature = vgg16_bn(True)
        self.im2col = Im2col(3, 1, 1)
        self.lstm = BLSTM(3*3*512, 128, True, True)

        self.fc = nn.Conv2d(256, 512, 3, 1)

        self.cls = nn.Conv2d(512, 2*10, kernel_size=1, stride=1)
        self.loc = nn.Conv2d(512, 2*10, kernel_size=1, stride=1)
        # self.side_refinement = nn.Conv2d(512, 10, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.feature(x)
        x = self.im2col(x)
        x = self.lstm(x)
        x = self.fc(x)
        x = self.relu(x)

        score = self.cls(x) # (N, 20, H, W)
        loc = self.loc(x) # (N, 20, H, W)
        # side_refinement = self.side_refinement(x)

        return score, loc


if __name__ == '__main__':
    model = CTPN()
    print(model)

