import SinkhornDRO
import DataProcessorLogistic
import LinearERM
import fDRO
import sinkhorn_base
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
torch.manual_seed(42)



def sinkhorn_train(model, lr ,eta0_list,train_dataloader, num_epochs, checkpoint_dir, interval=5):
    global device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.long()
            num_zeta = batch_data.size(0)
            batch_data.to(device)
            batch_label.to(device)
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
            checkpoint_path = f"{checkpoint_dir}/DRO_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(outer_loss_list)
    return outer_loss_list


def sinkhorn_wang_train(model, lr ,train_dataloader, num_epochs, checkpoint_dir, interval=5):
    global device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.long()
            num_zeta = batch_data.size(0)
            batch_data.to(device)
            batch_label.to(device)
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
    
        
def fDRO_train(model, lr ,train_dataloader, num_epochs, checkpoint_dir,eta, interval=5):
    global device
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
            batch_data.to(device)
            batch_label.to(device)
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

def ERM_train(model,lr ,train_dataloader, num_epochs, checkpoint_dir, interval = 5):
    global device
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
            checkpoint_path = f"{checkpoint_dir}/Linear_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(train_loss_list)
    return train_loss_list
    



if __name__ == "__main__":
    """
    Hyper-parameter setting for dataset 
    (You should NOT change synethic dataset unless it's too hard to converge.)
    """
    # Path to the saved file
    device = torch.device("cpu")
    file_path = "cifar10_resnet_features.pth"
    
    SEED = 2022
    torch.manual_seed(SEED)
    
    # Load the features
    data = torch.load(file_path)
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]
    #print(train_labels.shape)
    
    """
    sub-sample test dataset
    """
    
    num_test_samples = 1000

    # Ensure you don't exceed the size of the dataset
    num_test_samples = min(num_test_samples, test_features.size(0))

    # Randomly sample indices
    indices = torch.randperm(test_features.size(0))[:num_test_samples]

    # Subsample features and labels
    sub_sampled_test_features = test_features[indices]
    sub_sampled_test_labels = test_labels[indices]
    print('sub sampled label size is',sub_sampled_test_labels.shape)
    
    BATCH_SIZE = 64
    EVAL_SIZE = 256
    num_epochs = 80
    """
    ----- Data Generation
    """
    
    
    #X_train = dataset[:, :-1]
    #y_train = dataset[:, -1]
    #y_train = y_train.unsqueeze(1)
    #print(dataset.size())
    #print(X_train.size())
    #print(y_train.size())
    ## represents x^*
    """
    ---- Model parameter Initilization
    """
    n = train_features.size(1)
    #print(n)
    m = train_features.size(0)
    # Intialize primal and dual variable.
    weights_primal = torch.empty((10,n))  # Replace 10 with the size of your parameter

    # Compute fan_in manually (assuming input features = size of weights)
    fan_in = weights_primal.size(0)
    std = math.sqrt(0.2)
    x0=torch.nn.init.normal_(weights_primal, mean=0.0, std=std)
    x0 = x0.requires_grad_(True)
    #print(x0.shape)
    # in-context dual variable initialization
    weights_dual = torch.empty(m)
    fan_in = weights_dual.size(0)
    std = math.sqrt(0.1)
    eta0_initialize = torch.nn.init.normal(weights_dual, mean = 1.0, std = std)
    #std_eta0 = eta0_initialize.std()
    #eta0_initialize = (eta0_initialize - mean_eta0)/std_eta0
    eta0_list = eta0_initialize.clone().detach()


    x0 = x0.to(device)
    #x0 = x0.view(1,-1)
    #print(x0.shape)
    eta0_list = eta0_list.to(device)
    #X_train = X_train.to(device)
    #y_train = y_train.to(device)

    """
    ---- Sinkhorn DRO and other baseline model Initilaization.
    """

    loss_type = 'classification'
    model_DRO = SinkhornDRO.SinkhornDRO(n,10,x0.clone().detach())
    model_DRO.num_xi = 2
    #num_xi = model_DRO.num_xi
    model_DRO.loss_type = loss_type
    model_DRO.regularization = 0.8
    model_DRO.epsilon = 1.0
    model_DRO.inner_loop = 5
    model_DRO.lr1 = 8e-2
    model_DRO.lr2 = 1e-1
    model_DRO.to(device)
    #inputd,outputd = model.generate_corrput_data(X_train[0,:], y_train[0,:])
    #print(inputd.size())
    #print(outputd.size())
    
    """
    sinkhorn_wang
    """
    loss_type = 'classification'
    model_base_DRO = sinkhorn_base.WangSinkhorn(n,10,x0.clone().detach())
    model_base_DRO.num_xi = 2
    #num_xi = model_DRO.num_xi
    model_base_DRO.loss_type = loss_type
    model_base_DRO.regularization = 0.8
    model_base_DRO.epsilon = 1.0
    model_base_DRO.lr1 = 8e-2
    model_base_DRO.to(device)
    
    
    """
    f-DRO initialization
    """
    model_fDRO = fDRO.fDRO(n, 10, x0.clone().detach())
    model_fDRO.loss_type = loss_type
    model_fDRO.lr1 = 8e-2
    model_fDRO.regularization = 0.8
    model_fDRO.to(device)
    eta_fDRO = 1.5
    interval = 10
    """
    ERM initialization
    """
    model_ERM = LinearERM.LinearModel(n, 10, x0.clone().detach())
    ERM_lr = 3e-2
    model_ERM.to(device)
    """
    train starts here
    """
    checkpoint_dir =  r"C:\Users\ynyang94\OneDrive - Texas A&M University\Documents\checkpoints_logi"
    
    train_dataset = torch.cat((train_features, train_labels.unsqueeze(1)), dim=1)
    train_data = DataProcessorLogistic.CustomDataset(train_dataset)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,num_workers = 8,shuffle=False, pin_memory=True)
    #eval_dataloader = DataLoader(train_dataset, batch_size=EVAL_SIZE,num_workers = 8 ,shuffle=False, pin_memory=True)

    # address need to be changed.
    
    # only change ERM learning rate.
    #outer_loss_list = sinkhorn_train(model_DRO,model_DRO.lr1,eta0_list.clone().detach(),train_dataloader, num_epochs, checkpoint_dir, interval)
    outer_loss_list_wang_DRO = sinkhorn_wang_train(model_base_DRO, model_base_DRO.lr1, train_dataloader, num_epochs, checkpoint_dir,interval)
    #outer_loss_list_fDRO = fDRO_train(model_fDRO, model_fDRO.lr1, train_dataloader, num_epochs, checkpoint_dir, eta_fDRO, interval)
    #outer_loss_list_linear = ERM_train(model_ERM,ERM_lr, train_dataloader, num_epochs, checkpoint_dir, interval)
    
 