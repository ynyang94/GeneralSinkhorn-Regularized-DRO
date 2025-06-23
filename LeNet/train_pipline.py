# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 16:59:00 2025

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
from torch.utils.data import DataLoader
#import torchvision.transforms as T
# from torchvision.io import read_image
# from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import TensorDataset, DataLoader
import copy
import os
# import cvxpy as cp
import pdb
import math
import re

def sinkhorn_train(model, lr ,eta0_list,train_dataloader, num_epochs, checkpoint_dir, interval=5, device = 'cpu'):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.long()
            num_zeta = batch_data.size(0)
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            #print(batch_data.size(), batch_label.size())
            expected_loss = 0
            for i in range(num_zeta):
                # generate corrput data
                inputd,outputd = model.generate_corrput_data(batch_data[i,:], batch_label[i,])
                inputd = inputd.to(device)
                outputd = outputd.squeeze(dim=1)
                outputd = outputd.to(device)
                #print(inputd.size(), outputd.size())
                #predictions = torch.empty((num_xi,1))
                # generate predictions.
                #for j in range(num_xi):
                predictions = model(inputd)
                #print(predictions.size())
                predictions = predictions.to(device)
                #print(predictions.size())
                eta_index = idx[i]
                # generate inner_loss and optimize eta
                eta0_list[eta_index] = model.inner_minimization(predictions, eta0_list[eta_index], batch_data[i,:], inputd, outputd)
                inner_loss,_ = model.inner_objective(predictions, eta0_list[eta_index], batch_data[i,:], inputd, outputd)
                #inner_grad=model.gradient_eta(predictions, eta, X_train[i,:], y_train[i,:])
                expected_loss += inner_loss
            expected_loss = expected_loss/num_zeta
            #print(outer_loss)
            optimizer.zero_grad()  # Clear gradients from the last step
            expected_loss.backward()
            optimizer.step()
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print(f"Gradient of {name}: {param.grad}")
        #epoch_outer_loss += expected_loss.detach()
        outer_loss_list.append(expected_loss.detach())
        
        if epoch%interval == 0:
            checkpoint_path = f"{checkpoint_dir}/SDRO_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(outer_loss_list)
    return outer_loss_list


def sinkhorn_wang_train(model, lr ,train_dataloader, num_epochs, checkpoint_dir, interval=5, device = 'cpu'):
    #global device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.long()
            num_zeta = batch_data.size(0)
            batch_data= batch_data.to(device)
            batch_label = batch_label.to(device)
            #print(batch_data.size(), batch_label.size())
            expected_loss = 0
            for i in range(num_zeta):
                # generate corrput data
                inputd,outputd = model.generate_corrput_data(batch_data[i,:], batch_label[i,])
                inputd = inputd.to(device)
                outputd = outputd.squeeze(dim=1)
                outputd = outputd.to(device)
                #print(inputd.size(), outputd.size())
                #predictions = torch.empty((num_xi,1))
                # generate predictions.
                #for j in range(num_xi):
                predictions = model(inputd)
                #print(predictions.size())
                predictions = predictions.to(device)
                #print(predictions.size())
                inner_loss = model.baseline_sinkDRO(predictions, batch_data[i,:], inputd, outputd)
                #inner_grad=model.gradient_eta(predictions, eta, X_train[i,:], y_train[i,:])
                expected_loss += inner_loss
            expected_loss = (model.regularization*model.epsilon)*(expected_loss/num_zeta)
            #print(outer_loss)
            optimizer.zero_grad()  # Clear gradients from the last step
            expected_loss.backward()
            optimizer.step()
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print(f"Gradient of {name}: {param.grad}")
        #epoch_outer_loss += expected_loss.detach()
        outer_loss_list.append(expected_loss.detach())
    
        
        if epoch%interval == 0:
            checkpoint_path = f"{checkpoint_dir}/wangDRO_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(outer_loss_list)
    return outer_loss_list
        
def fDRO_train(model, lr ,train_dataloader, num_epochs, checkpoint_dir,eta, interval=5, device = 'cpu'):
    #global device
    eta = torch.tensor(eta, requires_grad = True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        expected_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.long()
            num_zeta = batch_data.size(0)
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            #print(batch_data.size(), batch_label.size())
            predictions = model(batch_data)
            predictions = predictions.to(device)
            #print(predictions.size())
            #eta_index = idx[i]
            # generate inner_loss and optimize eta
            #eta0_list[eta_index] = model.inner_minimization(predictions, eta0_list[eta_index], batch_data[i,:], inputd, outputd)
            fDRO_loss = model.baseline_fDRO(predictions, eta, batch_label)
            expected_loss += fDRO_loss
            #print(outer_loss)
            optimizer.zero_grad()  # Clear gradients from the last step
            fDRO_loss.backward()
            optimizer.step()
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print(f"Gradient of {name}: {param.grad}")
        #epoch_outer_loss += expected_loss.detach()
        expected_loss = expected_loss/len(train_dataloader)
        outer_loss_list.append(expected_loss.detach())
        
        if epoch%interval == 0:
            checkpoint_path = f"{checkpoint_dir}/fDRO_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(outer_loss_list)
    return outer_loss_list

def ERM_train(model,lr ,train_dataloader, num_epochs, checkpoint_dir, interval = 5, device = 'cpu'):
    #global device
    train_loss_list = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(interval, num_epochs+interval, interval):
        model.train()
        expected_train_loss = 0
        for idx, train_data, train_label in train_dataloader:
            #train_label = train_label.unsqueeze(1)
            #print(train_label)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            predictions = model(train_data)
            #print("shape is",predictions.shape)
            predictions = predictions.to(device)
            train_loss = model.cross_entropy_metric(predictions, train_label.long())
            #print(train_loss)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            expected_train_loss += train_loss
        expected_train_loss = expected_train_loss/len(train_dataloader)
        train_loss_list.append(expected_train_loss.detach())

        
        if epoch%interval == 0:
            checkpoint_path = f"{checkpoint_dir}/ERM_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(train_loss_list)
    return train_loss_list


