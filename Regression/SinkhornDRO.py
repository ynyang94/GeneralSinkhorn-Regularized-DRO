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

# Create a Linear model: Can be either Linear Regression or Losgistic Regression.


## Create Sinkhorn DRO class for constructing objective function.
class SinkhornDRO(nn.Module):
    def __init__(self, input_dim, output_dim, weight):
    # Primal variable weight aka x in paper should be in size (1,n)
    # set default variable: learning rate; stopping criteria; maximal iteration.
        super(SinkhornDRO, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias = False)
        self.linear.weight = nn.Parameter(weight)
        self.lr1 = 1e-3
        self.lr2 = 1e-1
        # Regularized Primal Problem.
        self.regularization = 1
        self.stop_criteria = 1e-5
        self.outer_loop = 10
        self.inner_loop = 10
        # Sinkhorn distance regularization
        self.epsilon = 1
        self.noise = 0.2
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
    
    def set_inner_loop(self, inner_loop):
        self.inner_loop = inner_loop
    
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
    
    def set_num_xi(self,num_xi):
        self.num_xi = num_xi

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
        
    
 
    
    def inner_objective(self, predictions, eta, per_zeta , generated_xi, generated_target):
        """
        

        Parameters
        ----------
        predictions: should be a group of predictions, the number should be same as
        xi.
        x : model parameters
        eta : dual variable.
        per_zeta : each zeta_sample.
        per_target : output y
        generated_xi: the xi samples generated by user(distribution shift).
        generated_target: the corresponding generated target by user.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        inner_expectation : objective value of inner problem.
        grad: gradient w.r.t dual variable eta
        
        """
        
        
        #m0 = per_zeta.shape[0]
        #x = nn.Parameter(x)
        eta = eta.detach()
        #per_target = per_target.reshape((1,1))
        #n = per_zeta.size(0)
        #per_zeta = per_zeta.reshape((1,n))
        xi_samples = generated_xi
        per_target = generated_target
        #xi_samples = xi_samples.to(self.device)
        #per_target = per_target.to(self.device)
        inner_values = torch.empty((self.num_xi,))
        inner_grad = torch.empty((self.num_xi,))
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
            quad_term = torch.div((loss_value - self.regularization*norm_diff - eta) 
                                 , (self.regularization * self.epsilon))
            quad_value = 0.25*(torch.clamp(quad_term+2,min=0.0).pow(2))
            #exp_term_4_grad = exp_term.clone().detach()
            quad_term_obj = (self.regularization*self.epsilon)*quad_value
            #print(exp_term_obj.size())
            inner_values[i] = quad_term_obj
            quad_grad = 0.5*torch.clamp(quad_term+2,min=0.0)
            inner_grad[i] = quad_grad.detach().item()
            
            #inner_grad.append(exp_term_4_grad)

        # Mean of inner values (sample mean approximation of inner expectation)
        # apply full gradient descent in inner optimization.
        inner_expectation = inner_values.mean() + eta
        
        grad = 1.0 - inner_grad.mean()
        #grad = 1.0 - torch.mean(torch.stack(inner_grad))
        return inner_expectation, grad
    
    # Not used anymore
    def gradient_eta(self, predictions, eta,per_zeta, generated_xi, generated_target):
        predictions = predictions.detach()
        predictions.to(self.device)
        #m0 = per_zeta.shape[0]
        eta = eta.detach()
        #per_zeta = per_zeta.reshape((1,n))
        #x = x.reshape((1,n))
        xi_samples = generated_xi
        per_target = generated_target
        #xi_samples = xi_samples.to(self.device)
        #per_target = per_target.to(self.device)
        #inner_values = []
        inner_grad = torch.empty((self.num_xi,))
        for i in range(self.num_xi):
            xi = xi_samples[i,:].unsqueeze(0)
            target = per_target[i]
            norm_diff = self.distance_metric(per_zeta.to(self.device),xi.to(self.device))
            prediction = predictions[i]
            # Select loss function
            
            #with torch.no_grad():
            
            #model.linear.weight =  nn.Parameter(x)
            # now assume bias = 0.
            #model.linear.bias = nn.Parameter(torch.tensor([0.0]))
            if self.loss_type == 'regression':
                loss_value = self.mse_metric(prediction, target)
            elif self.loss_type == 'classification':
                loss_value = self.cross_entropy_metric(prediction,target)
            else:
                raise ValueError("Invalid loss type. Choose 'regression' or 'classification'.")
            exp_term = 0.5*torch.clamp((loss_value.detach() - self.regularization*norm_diff - eta) / (self.regularization * self.epsilon)+2,min=0.0)
            #exp_term_4_grad = exp_term.clone().detach()
            #exp_term_obj = torch.multiply(self.regularization,self.epsilon)*exp_term
            #inner_values.append(exp_term_obj)
            inner_grad[i] = exp_term
        grad = 1.0 - inner_grad.mean()
        return grad
    
    def inner_minimization(self,predictions,eta, per_zeta, generated_xi, generated_target):
        """
        

        Parameters
        ----------
        x : primal variable aka model parameter (fixed).(should be passed by required_grad = False)
        eta : dual variable (in-contex), subject to change with zeta.
        per_zeta : data sample following nominal distribution P.
        per_target : output data y

        Returns
        -------
        eta : the optimized eta subject to each sample zeta.
        
        """
        #x = x.clone().detach()
        _,inner_grad = self.inner_objective(predictions, eta, per_zeta, generated_xi, generated_target)
        for i in range(self.inner_loop):
            eta = eta - self.lr2*inner_grad
            _,inner_grad = self.inner_objective(predictions, eta, per_zeta, generated_xi, generated_target)
        return eta.detach().item()
    
   
    def outer_objective(self, predictions, eta_list,zeta_samples,targets):
        """
        

        Parameters
        ----------
        x : model parameters
        eta_list : a tensor, since eta is in-context variable w.r.t zeta
        zeta_samples : mini-batch group sample of zeta
        targets : y

        Returns: outer objective function.
        -------

        """
        
        #x = nn.Parameter(x)
        
        m0 = zeta_samples.shape[0]
        m1 = eta_list.shape[0]
        if m0 != m1:
            raise ValueError("the number of eta list and zeta samples must be same")
        else:
            inner_expectations = torch.empty((m0,))
            for i in range(m0):
                zeta = zeta_samples[i,:]
                eta = eta_list[i].detach()
                per_target = targets[i,:]
                inner_expectations[i],_ = self.inner_objective(predictions, eta , zeta, per_target)
            outer_expectation = inner_expectations.mean()
        return outer_expectation
    
    def one_step_outer_minimization(self, x, eta_list, zeta, targets):
        """
        

        Parameters
        ----------
        x : primal variable to be optimized
            
        eta : dual variable
        zeta : a mini-batch of zeta
        targets : output y

        Returns
        -------
        x: the optimized x, should be a tensor with size n with one step progress

        """
        
        #x.requires_grad = True
        x = nn.Parameter(x)
        x_clone = x.clone().detach()
        eta_list = eta_list.detach()
        m = zeta.size(0)
        n = zeta.size(1)
        optimizer =torch.optim.SGD([x],lr = self.lr1)
        for j in range(m):
            per_zeta = zeta[j,:]
            per_target = targets[j,:]
            eta_initialize = eta_list[j]
            eta = self.inner_minimization(x_clone, eta_initialize, per_zeta, per_target)
            eta_list[j] = eta.detach()
            
        outer_objective = self.outer_objective(x, eta_list, zeta, targets)
        #outer_objective.requires_grad = True
        torch.autograd.set_detect_anomaly(True)
        outer_objective.backward()
        print("Gradients of x before step:", x.grad)
        optimizer.step()
        optimizer.zero_grad()
            
        
        return x
    
        
        
    

    
            
    #def single_dual(self,x,eta,xi,target,zeta):
        #input_dim = xi.shape(1)
        #output_dim = 1
        #model = LinearModel(input_dim, output_dim)
        #prediction = model(x)
        #if self.loss_type == 'regression':
            #original_loss = self.mse_metric(prediction, target)
        #if self.loss_type == 'classification':
            #original_loss = self.cross_entropy_metric(prediction, target)
        #distance_val = self.distance_metric(self,xi,zeta)
        #loss_input = (original_loss - torch.multiply(self.regularization, distance_val)-eta)/torch.multiply(self.regularization,self.epsilon)
        #loss_val = torch.exp(loss_input)
        #loss_val *= torch.multiply(self.regularization,self.epsilon)
        #loss_val += loss_val+eta
        #return loss_val
    
    #def expected_loss(self,x,eta,X_train,X_nominal,Y_train):
        #m = X_train.size(dim=0)
        #n = X_train.size(dim=1)
        #output_loss = []
        #for j in range(m):
            #zeta = X_nominal[j,:]
            #inner_loss=[]
            #for i in range(m):
                #xi = X_train[i,:]
                #target = Y_train[i]
                #loss_val = self.single_dual(x,eta,xi,target,zeta,self.loss_type)
                #inner_loss.append(loss_val)
            #output_loss.append(torch.tensor(inner_loss).mean())
        #output_loss = torch.tensor(output_loss).mean()
        #return output_loss
            
    
    # def outer_objective(self,x,eta,zeta_samples,targets):
    #     """
    #     # parameters:
    #         x: model parameter
    #         eta: dual variable
    #         zeta_samples: samples following nominal distribution P.
    #         xi_samples: samples following underleying distribution Q. 
    #         need to generate by ourselves.
    #         targets: output value y.
            
    #     # Output:
    #         returns objective value.

    #     """
    #     m0 = zeta_samples.shape[0]
    #     n = zeta_samples.shape[1]
    #     inner_expectations = []
    #     x = x.reshape((1,n))
    #     for j in range(m0):
    #         zeta = zeta_samples[j,:]
    #         xi_samples = []
    #         for i in range(self.num_xi):
    #             distribution_shift = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(self.noise)), size=zeta.size())
    #             xi_sample = zeta+distribution_shift
    #             xi_samples.append(xi_sample)
            
    #         inner_values = []
    #         for i in range(self.num_xi):
    #             xi = xi_samples[i,:].unsqueeze(0)
    #             target = targets[i].unsqueeze(0)
    #             norm_diff = self.distance_metric(zeta,xi)
    #             # Select loss function
    #             model = LinearModel(n,1)
    #             #with torch.no_grad():
    #             model.linear.weight =  nn.Parameter(x)
    #             # now assume bias = 0.
    #             model.linear.bias = nn.Parameter(torch.tensor([0.0]))
    #             prediction = model(xi)
    #             if self.loss_type == 'regression':
    #                 loss_value = self.mse_metric(prediction,target)
    #                 #print(loss_value)
    #             elif self.loss_type == 'classification':
    #                 loss_value = self.cross_entropy_metric(prediction,target)
    #             else:
    #                 raise ValueError("Invalid loss type. Choose 'regression' or 'classification'.")

    #             # Compute the inner term
    #             exp_term = torch.exp((loss_value - self.regularization*norm_diff - eta) / (self.regularization * self.epsilon)-1)
    #             exp_term *= torch.multiply(self.regularization,self.epsilon)
    #             inner_values.append(exp_term)

    #         # Mean of inner values (sample mean approximation of inner expectation)
    #         inner_expectation = torch.mean(torch.stack(inner_values))+ eta
    #         inner_expectations.append(inner_expectation)

    #     # Outer expectation (sample mean approximation over zeta samples)
    #     outer_expectation = torch.mean(torch.stack(inner_expectations))
    #     return outer_expectation
    #def optimizer(self,x,eta):
        #optimizer_x = torch.optim.SGD([x], lr=self.lr1)
        #optimizer_eta = torch.optim.SGD([eta], lr=self.lr2)
    
    #def inner_minimization(self, x, eta, per_zeta, per_target):
       
        #x_without_grad = x.detach()
        #eta = eta.detach().clone().requires_grad_(True)
        
        #for i in range(self.inner_loop):
            #_,inner_grad = self.inner_minimization(x_without_grad, eta, per_zeta,per_target) # Calculate loss
            #eta = eta - self.lr2*inner_grad
       # return eta


    




    


    
    
    
