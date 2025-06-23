# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:28:49 2025

@author: ynyang94
"""

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

class WangSinkhornDRO_LeNet(nn.Module):
    def __init__(self, input_dim=(1, 28, 28), num_classes=10):
    # Primal variable weight aka x in paper should be in size (1,n)
    # set default variable: learning rate; stopping criteria; maximal iteration.
        super(WangSinkhornDRO_LeNet, self).__init__()
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Define layers
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
        
        self.lr1 = 1e-3
        # Regularized Primal Problem.
        self.regularization = 1
        self.stop_criteria = 1e-5
        # Sinkhorn distance regularization
        self.epsilon = 1
        self.noise = 0.15
        self.num_xi = 1
        self.loss_type = 'classification'
       
        # initialize x and \eta
        #self.x = torch.randn(10, requires_grad=True)  # Example 10-dimensional vector
        #self.eta = torch.tensor(1.0, requires_grad=True)  # Scalar eta
    
    def _initialize_weights(self):
        """
        Initializes weights for the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    
    # Forward Pass
    def forward(self, xi):
        xi = F.relu(self.conv1(xi))
        xi = self.pool(xi)
        xi = F.relu(self.conv2(xi))
        xi = self.pool(xi)
        xi = torch.flatten(xi, 1)  # Flatten for fully connected layers
        xi = F.relu(self.fc1(xi))
        xi = F.relu(self.fc2(xi))
        xi = self.fc3(xi)
        return xi
    
    def set_outer_loop(self,outer_loop):
        self.outer_loop = outer_loop
    
    def set_lr1(self, learning_rate):
        self.lr1 = learning_rate
    
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def set_threshold(self,stop_criteria):
        self.stop_criteria = stop_criteria
    
    def set_noise(self,noise):
        self.noise = noise
    
    def generate_corrput_data(self,per_zeta, per_target):
        """
        
        This function generates training examples following underlying distribution Q.
        This function assumes there is no distribution shift over target, aka y.

        """
        device = self.device
        per_zeta = per_zeta.to(device)
        per_target = per_target.to(device)
        per_zeta = per_zeta.unsqueeze(0)
        #torch.manual_seed(2020)
        n, c, h, w = per_zeta.size()
        per_target = per_target.repeat(self.num_xi, 1)
        per_zeta = per_zeta.repeat(self.num_xi, 1, 1, 1)
        # Generate Gaussian noise for each sample
        # Noise shape must match the per_zeta shape
        
        distribution_shift = torch.randn((self.num_xi * n, c, h, w), device=device) * self.noise

        # Add Gaussian noise to the images
        xi_samples = per_zeta + distribution_shift
        xi_samples = xi_samples.to(device)
        per_target = per_target.to(device)
        #print("Shape of xi_samples:", xi_samples.shape)

        return xi_samples, per_target
    
    
    # for c(\zeta,\xi).
    def distance_metric(self,xi,zeta):
        return torch.norm(xi-zeta,p=2)
    
    # for different loss function ell.
    def mse_metric(self, predictions, targets):
        #x = x.requires_grad_(True)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #n = x.size(1)
        #model = LinearModel(n,1,x)
        #model = model.to(device)
        #with torch.no_grad():
        
        #model.linear.weight =  nn.Parameter(x)
        # now assume bias = 0.
        #model.linear.bias = nn.Parameter(torch.tensor([0.0]))
        #predictions = model(inputs)
        return F.mse_loss(predictions, targets, reduction='mean')#.item()

    # Define Cross-Entropy metric
    def cross_entropy_metric(self,predictions, targets):
        return F.cross_entropy(predictions, targets, reduction='mean')#.item()
    
    def baseline_sinkDRO(self, predictions, per_zeta , generated_xi, generated_target):
        """
        

        Wang, Jie, Gao, Rui, Xie, Yao Sinkhorn DRO baseline

        """
        xi_samples = generated_xi
        per_target = generated_target
        inner_values = torch.empty((self.num_xi,))
        
        for i in range(self.num_xi):
                xi = xi_samples[i,:].unsqueeze(0)
                target = per_target[i]
                norm_diff = self.distance_metric(per_zeta.to(self.device),xi.to(self.device))
                prediction = predictions[i]
                # Select loss function
                #loss = nn.MSELoss()
                prediction = prediction.to(self.device)
                if self.loss_type == 'regression':
                    loss_value = self.mse_metric(prediction, target)
                elif self.loss_type == 'classification':
                    loss_value = self.cross_entropy_metric(prediction,target)
                else:
                    raise ValueError("Invalid loss type. Choose 'regression' or 'classification'.")
                    #print(loss_value)
                    # Compute the inner term
                inner_term = torch.div((loss_value - self.regularization*norm_diff), 
                                  (self.regularization * self.epsilon))
                #quad_value = 0.25*(torch.clamp(quad_term+2,min=0.0).pow(2))
                exp_term = torch.exp(inner_term)
                #exp_term_obj = (self.regularization*self.epsilon)*exp_term
                #print(exp_term_obj.size())
                inner_values[i] = exp_term
            
                #inner_grad.append(exp_term_4_grad)

                # Mean of inner values (sample mean approximation of inner expectation)
                # apply full gradient descent in inner optimization.
                inner_expectation = inner_values.mean()
                
    
        return torch.log(inner_expectation)