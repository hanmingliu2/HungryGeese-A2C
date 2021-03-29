# -*- coding: utf-8 -*-
"""
@author: Hanming Liu
"""


from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size,
                              padding=kernel_size//2)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)

        
    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size)
        
        
    def forward(self, x):
        return F.relu(x + self.conv2(self.conv1(x)))
    
    
# A small ResNet with 10 residual blocks of 64 convolutional filters each.
# An extra average pooling layer is used to reduce model size.
# It outputs both a action policy and a state value.
class ZZGooseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(9, 64, 3)
        
        self.residuals = nn.ModuleList([
            ResidualBlock(64, 3) for _ in range(10)
        ])
        
        self.pooling = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        self.policy_head = nn.Linear(960, 4)
        self.value_head = nn.Linear(960, 1)
        
        
    def forward(self, x):
        x = self.conv(x)
        for block in self.residuals:
            x = block(x)
        
        x = self.pooling(x)
        x = self.flatten(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
    

def create_network(gpu=True, training=True):
    net = ZZGooseNet().cuda() if gpu else ZZGooseNet()
    if path.isfile('ZZGooseNet.pt'):
        net.load_state_dict(torch.load('ZZGooseNet.pt',
                                       map_location='cuda' if gpu else 'cpu'))
        
    if training:
        net.train()
    else:
        net.eval()
    return net
