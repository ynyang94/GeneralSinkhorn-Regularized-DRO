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
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, weight):
        super(LinearModel, self).__init__()
        self.device = torch.device("cpu")
        weight = weight.to(self.device)
        self.linear = nn.Linear(input_dim, output_dim, bias = False)
        self.linear.weight = nn.Parameter(weight)
        
    
    def forward(self, x):
        #x = x.to(self.device)
        return self.linear(x)
    
    def mse_metric(self, predictions, targets):
        return F.mse_loss(predictions.to(self.device), targets.to(self.device), reduction='mean')
    
    def cross_entropy_metric(self,predictions, targets):
        #print(predictions.dtype)
        return F.cross_entropy(predictions.to(self.device), targets.to(self.device), reduction='mean')#.item()
