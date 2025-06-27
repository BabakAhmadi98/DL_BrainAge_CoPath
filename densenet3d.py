#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Babak Ahmadi
3-D DenseNet-121 variant used in the manuscript, by Ahmadi et al.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import random
import os
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f'Number of available GPUs: {num_gpus}')

class Scale(nn.Module):
    def __init__(self, num_features):
        super(Scale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1, 1)
        return gamma * x + beta

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        inter_channels = in_channels * 4

        self.bn1 = nn.BatchNorm3d(in_channels, eps=1.1e-5, momentum=0.01)
        self.scale1 = Scale(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm3d(inter_channels, eps=1.1e-5, momentum=0.01)
        self.scale2 = Scale(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pad = nn.ConstantPad3d(1, 0)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=3, padding=0, bias=False)

        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.bn1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.bn2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.pad(out)
        out = self.conv2(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ConvBlock(in_channels + i * growth_rate, growth_rate, dropout_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, compression=1.0, dropout_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels, eps=1.1e-5, momentum=0.01)
        self.scale = Scale(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(int(in_channels * compression), int(in_channels * compression), kernel_size=1, bias=False)
        self.dropout_rate = dropout_rate
        self.pool = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.scale(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.pool(x)
        return x

class DenseNet3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, growth_rate=48, block_layers=[3,6,12,8],
                 num_init_features=64, reduction=0.0, dropout_rate=0.0):
        super(DenseNet3D, self).__init__()
        self.reduction = reduction
        self.compression = 1.0 - self.reduction

        self.pad1 = nn.ConstantPad3d(3, 0)  
        self.conv1 = nn.Conv3d(input_channels, num_init_features, kernel_size=5, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(num_init_features, eps=1.1e-5, momentum=0.01)
        self.scale1 = Scale(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.pad_pool = nn.ConstantPad3d(1, 0) 
        self.pool1 = nn.MaxPool3d(3, stride=2, padding=0)

        num_channels = num_init_features
        self.blocks = nn.ModuleList()

        # Dense blocks and transition blocks
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_channels, growth_rate, dropout_rate)
            self.blocks.append(block)
            num_channels = num_channels + num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionBlock(num_channels, num_channels, compression=self.compression, dropout_rate=dropout_rate)
                self.blocks.append(trans)
                num_channels = int(num_channels * self.compression)

        self.bn_final = nn.BatchNorm3d(num_channels, eps=1.1e-5, momentum=0.01)
        self.scale_final = Scale(num_channels)
        self.relu_final = nn.ReLU(inplace=True)
        self.CAM_conv = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(num_channels, num_classes)
        self.apply(weights_init)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.scale1(x)
        x = self.relu(x)
        x = self.pad_pool(x)
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.bn_final(x)
        x = self.scale_final(x)
        x = self.relu_final(x)
        x = self.CAM_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EarlyStopping:
    def __init__(self, patience=6, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0