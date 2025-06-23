# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:20:24 2025

@author: ynyang94
"""

"""
Created on Tue Jan 21 16:01:47 2025

@author: ivanyang
"""

import DeepSinkhorn
import DataProcessorLogistic
import DeepERM
import DeepfDRO
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
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import copy
import os
# import cvxpy as cp
import pdb
import math
import re
import test_and_attack as test

torch.manual_seed(42)




    







if __name__ == "__main__":
    """
    Hyper-parameter setting for dataset 
    (You should NOT change synethic dataset unless it's too hard to converge.)
    """
    # Path to the saved file
    device = torch.device("cpu")
    
    
    SEED = 42
    torch.manual_seed(SEED)
    
    # Load the features
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load 
    train_dataset = DataProcessorLogistic.CustomMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = DataProcessorLogistic.CustomMNIST(root='./data', train=False, download=True, transform=transform)


    #print(train_labels.shape)
    
    """
    sub-sample test dataset
    """
    
    num_test_samples = 1000

    # Ensure you don't exceed the size of the dataset
    num_test_samples = min(num_test_samples, len(test_dataset))

    # Randomly sample indices
    indices = torch.randperm(len(test_dataset))[:num_test_samples]

    # Subsample features and labels
    sub_sampled_test_data = Subset(test_dataset, indices)
    print('sub sampled label size is',len(sub_sampled_test_data))
    
    BATCH_SIZE = 128
    EVAL_SIZE = 128
    num_epochs = 100
    
    interval = 10
   
    """
    ---- Model parameter Initilization
    """
    m = len(train_dataset)
    # Intialize primal and dual variable.
    #weights_primal = torch.empty((10,n))  # Replace 10 with the size of your parameter

    # Compute fan_in manually (assuming input features = size of weights)
    #fan_in = weights_primal.size(0)
    #std = math.sqrt(0.2)
    #x0=torch.nn.init.normal_(weights_primal, mean=0.0, std=std)
    #x0 = x0.requires_grad_(True)
    #print(x0.shape)
    # in-context dual variable initialization
    weights_dual = torch.empty(m)
    fan_in = weights_dual.size(0)
    std = math.sqrt(0.1)
    eta0_initialize = torch.nn.init.normal(weights_dual, mean = 0.5, std = std)
    #std_eta0 = eta0_initialize.std()
    #eta0_initialize = (eta0_initialize - mean_eta0)/std_eta0
    eta0_list = eta0_initialize.clone().detach()


    #x0 = x0.to(device)
    #x0 = x0.view(1,-1)
    #print(x0.shape)
    eta0_list = eta0_list.to(device)
    #X_train = X_train.to(device)
    #y_train = y_train.to(device)
 
    """
    ---- Sinkhorn DRO and other baseline model Initilaization.
    """
 
    
    model_DRO = DeepSinkhorn.LeNetSinkhornDRO(input_dim = (3,32,32),num_classes=10)
    model_DRO.num_xi = 4
    num_xi = model_DRO.num_xi
    model_DRO.regularization = 0.5
    model_DRO.epsilon = 0.8
    model_DRO.inner_loop = 10
    model_DRO.lr1 = 5e-3
    model_DRO.lr2 = 1e-1
    model_DRO.noise = 0.15
    model_DRO = model_DRO.to(device)
    #inputd,outputd = model.generate_corrput_data(X_train[0,:], y_train[0,:])
    #print(inputd.size())
    #print(outputd.size())
    
    model_DRO_base = sinkhorn_base.WangSinkhornDRO_LeNet(input_dim = (3,32,32),num_classes=10)
    model_DRO_base.num_xi = 5
    model_DRO_base.regularization = 0.5
    model_DRO_base.epsilon = 0.8
    model_DRO_base.lr1 = 1e-3
    model_DRO_base.noise = 0.15
    model_DRO_base = model_DRO_base.to('cpu')
    
    
    """
    f-DRO initialization
    """
    model_fDRO = DeepfDRO.LeNetfDRO(input_dim = (3,32,32),num_classes=10)
    model_fDRO.lr1 = 1e-3
    model_fDRO.regularization = 0.5
    model_fDRO = model_fDRO.to(device)
    eta_fDRO = 1.0
    
    """
    ERM initialization
    """
    model_ERM = DeepERM.LeNetModel(input_dim = (3,32,32),num_classes=10)
    ERM_lr = 1e-3
    model_ERM = model_ERM.to(device)
    """
    train starts here
    """
    

    # address need to be changed.
    checkpoint_dir = r"C:\Users\ynyang94\OneDrive - Texas A&M University\Documents\checkpoints_DL"
    # only change ERM learning rate.
    
    """
     attack start here
    """
    attack_method = 'pgd2'
    epsilon = 1.2
    iteration= 30
    mu = 1.0
    
    #test_dataset_DRO = PGD_attack.generate_pgd_attacked_dataset(model_DRO, sub_sampled_test_data, epsilon, iteration)
    #test_dataset_ERM = PGD_attack.generate_pgd_attacked_dataset(model_ERM, sub_sampled_test_data, epsilon, iteration)
    #test_dataset_fDRO = PGD_attack.generate_pgd_attacked_dataset(model_fDRO, sub_sampled_test_data, epsilon, iteration)
    
    
    test_dataset_DRO_base = test.generate_pgd_attacked_dataset(model_DRO_base, sub_sampled_test_data, 
                                                                      checkpoint_dir,100,epsilon,iteration,EVAL_SIZE, 'wangDRO')
    test_dataset_ERM = test.generate_pgd_attacked_dataset(model_ERM, sub_sampled_test_data, 
                                                                      checkpoint_dir,100,epsilon,iteration, EVAL_SIZE, 'ERM')
    test_dataset_fDRO = test.generate_pgd_attacked_dataset(model_fDRO, sub_sampled_test_data, 
                                                                       checkpoint_dir,100,epsilon,iteration,EVAL_SIZE, 'fDRO')
    test_dataset_DRO = test.generate_pgd_attacked_dataset(model_DRO, sub_sampled_test_data, 
                                                          checkpoint_dir,100,epsilon,iteration, EVAL_SIZE,'SDRO')
           

    """
    fgsm attack here
    """                                                           
    #test_dataset_DRO = test.generate_fgsm_attacked_dataset(model_DRO, sub_sampled_test_data, 
    #                                                                  checkpoint_dir,num_epochs,epsilon, EVAL_SIZE,'SDRO')
    #test_dataset_DRO_base = test.generate_fgsm_attacked_dataset(model_DRO_base, sub_sampled_test_data, 
    #                                                                  checkpoint_dir,num_epochs,epsilon,EVAL_SIZE, 'wangDRO')
    #test_dataset_ERM = test.generate_fgsm_attacked_dataset(model_ERM, sub_sampled_test_data, 
    #                                                                  checkpoint_dir,num_epochs,epsilon, EVAL_SIZE, 'ERM')
    #test_dataset_fDRO = test.generate_fgsm_attacked_dataset(model_fDRO, sub_sampled_test_data, 
    #                                                                   checkpoint_dir,num_epochs,epsilon,EVAL_SIZE, 'fDRO')
    
    
    """
    mim attack here
    """
    #test_dataset_DRO_base = test.generate_mim_attacked_dataset(model_DRO_base, sub_sampled_test_data, 
    #                                                                  checkpoint_dir,100,epsilon,mu,iteration,EVAL_SIZE, 'wangDRO')
    #test_dataset_ERM = test.generate_mim_attacked_dataset(model_ERM, sub_sampled_test_data, 
    #                                                                  checkpoint_dir,100,epsilon,mu,iteration, EVAL_SIZE, 'ERM')
    #test_dataset_fDRO = test.generate_mim_attacked_dataset(model_fDRO, sub_sampled_test_data, 
    #                                                                   checkpoint_dir,100,epsilon,mu,iteration,EVAL_SIZE, 'fDRO')
    #test_dataset_DRO = test.generate_mim_attacked_dataset(model_DRO, sub_sampled_test_data, 
    #                                                      checkpoint_dir,100,epsilon,mu,iteration, EVAL_SIZE,'SDRO')
    
    
    #print(outer_loss_list)

    
    
    test_dataloader_SDRO = DataLoader(test_dataset_DRO, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_SDRO_base = DataLoader(test_dataset_DRO_base, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_ERM = DataLoader(test_dataset_ERM, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_fDRO = DataLoader(test_dataset_fDRO, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)


    """
    test for sinkhorn DRO
    """
    print("SDRO test")
    test_loss_list = test.test_or_validation(model_DRO,test_dataloader_SDRO, 
                                             checkpoint_dir, num_epochs, interval, train_method='SDRO')
    
    """
    test for SInkhorn DRO baseline
    """
    print("wang test")
    test_loss_list_base = test.test_or_validation(model_DRO_base,test_dataloader_SDRO_base, 
                                                  checkpoint_dir, num_epochs, interval, train_method ='wangDRO')
    
    """
    test for f-DRO
    """
    print("fDRO test")
    test_loss_list_fDRO = test.test_or_validation(model_fDRO,test_dataloader_fDRO, 
                                                  checkpoint_dir, num_epochs,interval, train_method = "fDRO")
    """
    test for linear ERM
    """
    print("ERM test")
    test_loss_list_linear = test.test_or_validation(model_ERM,
                                           test_dataloader_ERM, checkpoint_dir, num_epochs, interval, train_method = "ERM")
   
    
    
    
    #print("train loss linear",outer_loss_linear)
    
    test_loss_DRO = [loss.cpu().item() for loss in test_loss_list]
    
    test_loss_DRO_base = [loss.cpu().item() for loss in test_loss_list_base]
    
    test_loss_fDRO = [ loss.cpu().item() for loss in test_loss_list_fDRO]
    
    test_loss_linear = [ loss.cpu().item() for loss in test_loss_list_linear]
    """
    plot starts here
    """
    epochs_list = list(range(interval, num_epochs + interval, interval))

    #fig=plt.figure()
    #plt.loglog(epochs_list, outer_loss_DRO,'-o',color = 'red', label="SDRO")
    #plt.loglog(epochs_list, outer_loss_linear,'-+',color = 'blue', label = "ERM ")
    #plt.loglog(epochs_list, outer_loss_fDRO,'-*',color = 'green', label = "fDRO ")

    #plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    #plt.title("Train Loss Curve")
    #plt.legend(loc = 'lower left')
    #plt.show()
    #fig.savefig('train_LeNet.png')

    fig=plt.figure()
    plt.loglog(epochs_list, test_loss_DRO,'-o',color='red',label="SDRO2 "+f" {attack_method} {epsilon}")
    
    plt.loglog(epochs_list, test_loss_DRO_base,'-o',color='orange',label="SDRO1 "+f" {attack_method} {epsilon}")
    #plt.loglog(epochs_list, test_loss_DRO_1, '-o',color = 'red',label="SDRO " + r"p=0.1")
    #plt.loglog(epochs_list, test_loss_DRO_2,'-o', color = 'red',label="SDRO " + r"p=0.5")
    plt.loglog(epochs_list, test_loss_fDRO,'-*',color = 'green', label = "fDRO "+ f" {attack_method} {epsilon}")
    #plt.loglog(epochs_list, test_loss_fDRO_1,'-*', color = 'green',label = "fDRO "+ r"p=0.1")
    #plt.loglog(epochs_list, test_loss_fDRO_2,'-*', color = 'green',label = "fDRO "+ r"p=0.5")
    plt.loglog(epochs_list, test_loss_linear,'-+', color = 'blue',label = "ERM "+ f" {attack_method} {epsilon}")
    #plt.loglog(epochs_list, test_loss_linear_1,'-+', color = 'blue', label = "ERM "+ r"p=0.1")
    #plt.loglog(epochs_list, test_loss_linear_2,'-+', color = 'blue',label = "ERM "+ r"p=0.5")
    plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    plt.title("Test Loss Curve")
    plt.legend(loc ='lower left')
    plt.show()
    fig.savefig(f'test_LeNet_{attack_method}_{epsilon}.png')

    """
    hyper-parameters starts here.
    
    """

    python_file = "LeNet_train.py"

    # Output file
    output_file = "hyperparams_Dl.txt"
    # Define a pattern to capture variables and their values
    pattern = re.compile(r"^(\w+)\s*=\s*(.+)$")  # Captures "variable = value"


    with open(python_file, "r") as file, open(output_file, "w") as out_file:
        for line in file:
            # Strip comments and process only assignments
            line = line.split("#")[0].strip()
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                out_file.write(f"{key}: {value}\n")

 
    print(f"Hyper-parameters saved to {output_file}")