# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:04:29 2024

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import torchvision.transforms as T
# from torchvision.io import read_image
# from torchvision.models import resnet18, ResNet18_Weights
import copy
import os
# import cvxpy as cp
import pdb
import math

"""
This class is used to generate Synthetic Data for Linear Regression.
Hyper-parameters:
    num_samples
    input_dim: number of features
    mean, std: used for generating ground truth parameter x^*.
    multi_mean, covariance: used to generate train input and output.
    SEED: random seed.
    transform, target transform: any potential data preprocessing. default: None.
    
"""


class SyntheticDataset(Dataset):
    def __init__(self, num_samples, input_dim,mean,std, multi_mean, covariance, y_std, SEED, transform = None, target_transform = None):
        self.m = num_samples
        self.n = input_dim
        self.seed = SEED
        # for generation of ground_truth_parameter x^*
        self.mean = mean
        self.std =  std
        # for generation of Train (input and output)
        self.multi_mean = multi_mean
        self.convariance_matrix = covariance
        # used to define linear noise.
        self.y_std = y_std
        self.transform = transform
        self.target_transform = target_transform
        
        self.dataset = self.generate_data()
        #print(f"Dataset generated with shape: {self.dataset.shape}")
    
    def __len__(self):
        return self.m
    
    def generate_parameter(self):
        """
        

        Returns
        -------
        ground_truth_parameter : 
            optimal model parameter x^*

        """
        torch.manual_seed(self.seed)
        ground_truth_parameter = torch.normal(self.mean, self.std, size =(self.n,1))
        return ground_truth_parameter
    
    def generate_data(self):
        """
        

        Returns
        -------
        X_train : input training data.
        y_train : output training data.

        """
        torch.manual_seed(self.seed)
        #mean_multi = torch.ones((self.n,))
        #covariance_matrix = torch.eye(n)
        multi_normal_dist = torch.distributions.MultivariateNormal(self.multi_mean, self.convariance_matrix)
        X_train= multi_normal_dist.sample((self.m,))
        train_mean = X_train.mean(dim = 0, keepdim = True)
        train_std = X_train.std(dim = 0, keepdim = True)
        # normalization
        X_train = (X_train - train_mean)/train_std
        ground_truth_parameter = self.generate_parameter()
        # The model here is y = X^Tw+\epsilon, where \epsilon \sim Normal(0, y_std)
        y_train = torch.matmul(X_train,ground_truth_parameter)+torch.normal(0,self.y_std,(self.m,1))
        dataset = torch.cat([X_train, y_train], dim = 1)
        return dataset
        
    
    def __getitem__(self,idx):
        data = self.dataset[idx, :-1]  # All columns except the last
        label = self.dataset[idx, -1]  # Last column as label

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return idx, data, label
    
class PerturbedDataset(SyntheticDataset):
    def __init__(self, num_samples, input_dim,mean,std, multi_mean, 
                 covariance, y_std, SEED,perturbation_strength ,transform = None, target_transform = None):
        super().__init__(num_samples, input_dim,mean,std, multi_mean, 
                     covariance, y_std, SEED, transform = None, target_transform = None)
        self.perturbation_strength = perturbation_strength
    
    def __getitem__(self, idx):
        index, data, target = super().__getitem__(idx)
        perturbed_data = self.add_perturbation(data)
        return index, perturbed_data, target
    
    def add_perturbation(self, data):
        # Example perturbation: Add Gaussian noise
          laplace = torch.distributions.Laplace(loc=0.0, scale = self.perturbation_strength)
          laplace_noise = laplace.sample(data.shape).to(data.device)
          noise = self.perturbation_strength * laplace_noise
          #gaussian attack
          #noise = self.perturbation_strength*torch.rand_like(data)
          return data + noise
        
    
        