# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:41:17 2025

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

class WangSinkhorn(nn.Module):
    def __init__(self, input_dim, output_dim, weight):
    # Primal variable weight aka x in paper should be in size (1,n)
    # set default variable: learning rate; stopping criteria; maximal iteration.
        super(WangSinkhorn, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias = False)
        self.linear.weight = nn.Parameter(weight)
        self.lr1 = 1e-3
        # Regularized Primal Problem.
        self.regularization = 1
        self.stop_criteria = 1e-5
        self.outer_loop = 10
        # Sinkhorn distance regularization
        self.epsilon = 1
        self.noise = 0.1
        self.num_xi = 1
        self.loss_type = 'regression'
        self.device = torch.device("cpu")
        # initialize x and \eta
        #self.x = torch.randn(10, requires_grad=True)  # Example 10-dimensional vector
        #self.eta = torch.tensor(1.0, requires_grad=True)  # Scalar eta
    
    # Forward Pass
    def forward(self, xi):
        xi = xi.to(self.device)
        return self.linear(xi)
    
    def set_outer_loop(self,outer_loop):
        self.outer_loop = outer_loop
    
    def set_lr1(self, learning_rate):
        self.lr1 = learning_rate
    
    def set_lr2(self, learning_rate):
        self.lr2 = learning_rate
    
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
        #torch.manual_seed(2020)
        n = per_zeta.size(0)
        per_zeta = per_zeta.reshape((1,n))
        #xi_samples = torch.empty((self.num_xi,n))#.to(device)
        per_target = per_target.repeat(self.num_xi, 1)
        per_zeta = per_zeta.repeat(self.num_xi, 1)
        distribution_shift = torch.normal(mean=0.0, std=torch.tensor(self.noise), size=(self.num_xi,n))
        value = distribution_shift.clone()#.to(device)
        value = value.to(device)
        #for i in range(self.num_xi):
            #xi_sample = per_zeta+value[i,:]
            #xi_mean = xi_sample.mean()
            #xi_std = xi_sample.std()
            #xi_sample = (xi_sample -xi_mean)/xi_std
            #xi_samples[i,:] = xi_sample
            #targets[i,:] = per_target
        xi_samples = per_zeta+value
        xi_samples = xi_samples.to(self.device)
        per_target = per_target.to(device)
        return xi_samples, per_target
    
    #def initilization(self,x,eta):
        #self.x = x
        #self.eta = eta
    
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
                
                inner_expectation = inner_values.mean()
                
    
        return torch.log(inner_expectation)