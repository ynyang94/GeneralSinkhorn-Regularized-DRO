# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:57:16 2024

@author: ynyang94
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pandas import DataFrame
import csv
from fractions import Fraction
from pandas.plotting import scatter_matrix
import scipy.stats
from scipy.stats.mstats import winsorize
import random
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
#import torchvision.transforms as T
# from torchvision.io import read_image
# from torchvision.models import resnet18, ResNet18_Weights
import copy
import os
# import cvxpy as cp
import pdb
import math
import sys
sys.setrecursionlimit(2000)
"""
This class defines a linear model with ERM training paradigm.
"""
class LeNetModel(nn.Module):
    def __init__(self, input_dim=(1, 28, 28), num_classes=10):
        # input dim: should be a image
        # output dim: number of classes
        super(LeNetModel, self).__init__()
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate the size of the input to fc1
        dummy_input = torch.zeros(1, *input_dim)
        out = self.pool(nn.functional.relu(self.conv2(self.pool(nn.functional.relu(self.conv1(dummy_input))))))
        flattened_size = out.numel()
 
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Optionally initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initializes weights for the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def mse_metric(self, predictions, targets):
        return F.mse_loss(predictions.to(self.device), targets.to(self.device), reduction='mean')
    
    def cross_entropy_metric(self,predictions, targets):
        #print(predictions.dtype)
        return F.cross_entropy(predictions.to(self.device), targets.to(self.device), reduction='mean')#.item()
