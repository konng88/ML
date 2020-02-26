#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu, logsigmoid

class Highway(nn.Module):
    def __init__(self,in_features,out_features):
        super(Highway,self).__init__()
        self.gate = nn.Linear(in_features=in_features,out_features=out_features)
        self.proj = nn.Linear(in_features=in_features,out_features=out_features)

    def forward(self,X_conv_out):
        X_proj = relu(self.proj(X_conv_out))
        X_gate = logsigmoid(self.gate(X_conv_out))
        X_highway = X_gate * X_proj + (1-X_gate) * X_conv_out
        return X_highway

### END YOUR CODE
