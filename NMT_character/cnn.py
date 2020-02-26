#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
from torch.nn.functional import relu

class CNN(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=5, padding=0,dilation=1):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    def forward(self, X_input):
        conv_out = torch.max(relu(self.cnn(X_input)), 2)[0]
        return conv_out



### END YOUR CODE
#
# S, B, E, M, f = 30, 10, 5, 21, 8
# input = torch.randn(B,E,M)
# print(input.shape)
# c = CNN(E, f)
# out = c(input)
# print(out.shape)
