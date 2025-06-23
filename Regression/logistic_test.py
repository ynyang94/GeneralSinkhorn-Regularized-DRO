# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:34:39 2025

@author: ynyang94
"""

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

def test_or_validation(model, test_loader, checkpoint_dir, num_epochs, interval=5, train_method = "DRO"):
    global device
    test_loss_list = []
    acc_list = []
    with torch.no_grad():
        for epoch in range(interval, num_epochs + interval, interval):
            # Construct the checkpoint path based on the current epoch
            checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
        
            # Load the model's state dictionary from the checkpoint
            state_dict = torch.load(checkpoint_path, weights_only=True)  # Use weights_only=True here
            model.load_state_dict(state_dict)
            model.eval()  # Set model to evaluation mode

            expected_eval_loss = 0
            total_samples = 0 
            total_correct = 0
            with torch.no_grad():
                for _, data, label in test_loader:
                    #label = label.unsqueeze(1)
                    data = data.to(device)
                    label = label.to(device)
                    #num_zeta = data.size(0)
                    predictions = model(data)
                    predictions = predictions.to(device)
                    predicted_labels = predictions.argmax(dim=1)
                    # change here for classification task
                    test_loss = model.cross_entropy_metric(predictions, label.long())
                    expected_eval_loss += test_loss
                    # Get predicted class
                    total_correct += (predicted_labels == label).sum().item()
                    total_samples += label.size(0)
                expected_eval_loss = expected_eval_loss/len(test_loader)
                classification_accuracy = (total_correct/total_samples)*100
                
            test_loss_list.append(expected_eval_loss.detach())
            acc_list.append(classification_accuracy)
        print(test_loss_list)
        print(f"classification accuracy using {train_method} up to {epoch} is {acc_list}")
    return test_loss_list

def pgd_attack(model, images, labels, checkpoint_dir, epsilon, iters=40, epoch=80, train_method='DRO'):
    """
    Perform PGD attack under L-infinity norm.
    """
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    images = images.clone().detach()
    original_images = images.clone().detach()
    alpha = epsilon / 10

    # add random start within the epsilon ball
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, min=0, max=1).detach()

    for _ in range(iters):
        images = images.detach().requires_grad_()
        outputs = model(images)
        loss = model.cross_entropy_metric(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient sign update
        grad = images.grad.data
        adv_images = images + alpha * grad.sign()

        # Projection
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images


def sign_gradient_attack(model, inputs, labels, checkpoint_dir, epoch = 10,epsilon=0.01, train_method = 'DRO'):
    """
    Perform Sign Gradient Attack.
    Args:
        model: The model to attack.
        inputs: Original inputs / test dataloader.
        labels: True labels.
        epsilon: Perturbation strength.
    Returns:
        Perturbed inputs.
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    labels = labels.clone().detach()
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    # Set model to evaluation mode
    state_dict = torch.load(checkpoint_path, weights_only=True)  # Use weights_only=True here
    model.load_state_dict(state_dict)
    model.eval()

    # Forward pass
    outputs = model(inputs)
    #loss_fn = torch.nn.CrossEntropyLoss()  # Instantiate the loss function
    loss = model.cross_entropy_metric(outputs, labels)

    # Compute gradients
    loss.backward()

    # Generate perturbations
    perturbations = epsilon * inputs.grad.sign()

    # Apply perturbations
    perturbed_inputs = inputs + perturbations

    # Clip the values to ensure valid input range
    #perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    perturbed_inputs = perturbed_inputs.detach()

    return perturbed_inputs

if __name__ == "__main__":
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
    
    num_test_samples = 1500

    # Ensure you don't exceed the size of the dataset
    num_test_samples = min(num_test_samples, test_features.size(0))

    # Randomly sample indices
    indices = torch.randperm(test_features.size(0))[:num_test_samples]

    # Subsample features and labels
    sub_sampled_test_features = test_features[indices]
    sub_sampled_test_labels = test_labels[indices]
    print('sub sampled label size is',sub_sampled_test_labels.shape)
    
    BATCH_SIZE = 128
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
    
    checkpoint_dir =  r"C:\Users\ynyang94\OneDrive - Texas A&M University\Documents\checkpoints_logi"
    
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
    model_DRO.num_xi = 1
    num_xi = model_DRO.num_xi
    model_DRO.loss_type = loss_type
    model_DRO.regularization = 0.8
    model_DRO.epsilon = 1.0
    model_DRO.inner_loop = 8
    model_DRO.lr1 = 8e-2
    model_DRO.lr2 = 1e-1
    model_DRO.to(device)
    #inputd,outputd = model.generate_corrput_data(X_train[0,:], y_train[0,:])
    #print(inputd.size())
    #print(outputd.size())
    """
    -- Sinkhorn baseline from Wang, Jie
    """
    loss_type = 'classification'
    model_base_DRO = sinkhorn_base.WangSinkhorn(n,10,x0.clone().detach())
    model_base_DRO.num_xi = 2
    #num_xi = model_DRO.num_xi
    model_base_DRO.loss_type = loss_type
    model_base_DRO.regularization = 0.8
    model_base_DRO.epsilon = 1.0
    model_base_DRO.lr1 = 8e-3
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
    gradient sign attack start here
    """
    attack_method ="fgsm"
   
    epsilon = 0.00
   
    perturbed_features_DRO = sign_gradient_attack(model_DRO, sub_sampled_test_features,sub_sampled_test_labels,
                                                 checkpoint_dir,num_epochs,epsilon, train_method='DRO')
    perturbed_features_base_DRO = sign_gradient_attack(model_base_DRO, sub_sampled_test_features,sub_sampled_test_labels,
                                                 checkpoint_dir,num_epochs,epsilon, train_method='wangDRO')
    perturbed_features_ERM = sign_gradient_attack(model_ERM, sub_sampled_test_features, sub_sampled_test_labels,
                                                 checkpoint_dir, num_epochs, epsilon, train_method='Linear')
    perturbed_features_fDRO = sign_gradient_attack(model_fDRO, sub_sampled_test_features, sub_sampled_test_labels,
                                                  checkpoint_dir,num_epochs,epsilon, train_method = 'fDRO')
   
    """
    PGD atttack here (comment out when test sign_gradient attack)
    """
    #perturbed_features_DRO = pgd_attack(model_DRO, sub_sampled_test_features, sub_sampled_test_labels,
    #                                    checkpoint_dir,epsilon,20, num_epochs, train_method='DRO')
    #perturbed_features_base_DRO = pgd_attack(model_base_DRO, sub_sampled_test_features, sub_sampled_test_labels,
    #                                    checkpoint_dir,epsilon,20, num_epochs, train_method='wangDRO')
    #perturbed_features_ERM = pgd_attack(model_ERM, sub_sampled_test_features, sub_sampled_test_labels,
    #                                    checkpoint_dir, epsilon, 20, num_epochs, train_method = 'Linear')
    #perturbed_features_fDRO = pgd_attack(model_fDRO, sub_sampled_test_features, sub_sampled_test_labels,
    #                                     checkpoint_dir, epsilon, 20, num_epochs, train_method = 'fDRO')
   
    """
    Dataset Generation
    """
    perturbed_dataset_DRO = torch.cat((perturbed_features_DRO, sub_sampled_test_labels.unsqueeze(1)),dim = 1)
    #print(perturbed_dataset_DRO.shape)
    perturbed_dataset_base_DRO = torch.cat((perturbed_features_base_DRO, sub_sampled_test_labels.unsqueeze(1)),dim = 1)
    perturbed_dataset_ERM = torch.cat((perturbed_features_ERM, sub_sampled_test_labels.unsqueeze(1)),dim = 1)
    perturbed_dataset_fDRO = torch.cat((perturbed_features_fDRO, sub_sampled_test_labels.unsqueeze(1)),dim = 1)
   
    test_dataset_DRO = DataProcessorLogistic.CustomDataset(perturbed_dataset_DRO)
    test_dataset_base_DRO = DataProcessorLogistic.CustomDataset(perturbed_dataset_base_DRO)
    test_dataset_ERM = DataProcessorLogistic.CustomDataset(perturbed_dataset_ERM)
    test_dataset_fDRO = DataProcessorLogistic.CustomDataset(perturbed_dataset_fDRO)
   
   
    test_dataloader_SDRO = DataLoader(test_dataset_DRO, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    #for idx, data, label in test_dataloader_0:
    #    print('idx is', idx)
    #    print(data.shape)
    #    print(label.shape)
    test_dataloader_base_SDRO = DataLoader(test_dataset_base_DRO, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_ERM = DataLoader(test_dataset_ERM, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_fDRO = DataLoader(test_dataset_fDRO, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)


    """
    test for sinkhorn DRO
    """
    test_loss_list = test_or_validation(model_DRO,test_dataloader_SDRO, checkpoint_dir, num_epochs, interval)
    #test_loss_list_1 = test_or_validation(model_DRO,test_dataloader_1, checkpoint_dir, num_epochs, interval)
    #test_loss_list_2 = test_or_validation(model_DRO,test_dataloader_2, checkpoint_dir, num_epochs, interval)
    """
    test for base sinkhorn DRO
    """
    test_loss_list_base_DRO = test_or_validation(model_base_DRO,test_dataloader_base_SDRO, checkpoint_dir, num_epochs, interval, train_method = 'wangDRO')
    #test_loss_list_1 = test_or_validation(model_base_DRO,test_dataloader_1, checkpoint_dir, num_epochs, interval)
    #test_loss_list_2 = test_or_validation(model_base_DRO,test_dataloader_2, checkpoint_dir, num_epochs, interval)
   
    """
    test for f-DRO
    """
    test_loss_list_fDRO = test_or_validation(model_fDRO,test_dataloader_fDRO, checkpoint_dir, num_epochs,interval, train_method = "fDRO")
    #test_loss_list_fDRO_1 =  test_or_validation(model_fDRO,test_dataloader_1, checkpoint_dir, num_epochs,interval,train_method = "fDRO")
    #test_loss_list_fDRO_2 =  test_or_validation(model_fDRO,test_dataloader_2, checkpoint_dir, num_epochs, interval,train_method = "fDRO")
    """
    test for linear ERM
    """
    test_loss_list_linear = test_or_validation(model_ERM,
                                          test_dataloader_ERM, checkpoint_dir, num_epochs, interval, train_method = "Linear")
    #test_loss_list_linear_1 = test_or_validation(model_ERM,
    #                                       test_dataloader_1, checkpoint_dir, num_epochs, interval, train_method = "Linear")
    #test_loss_list_linear_2 = test_or_validation(model_ERM,
    #                                       test_dataloader_2, checkpoint_dir, num_epochs, interval, train_method = "Linear")
   
    #outer_loss_DRO = [loss.cpu().item() for loss in outer_loss_list]
    #outer_loss_fDRO = [loss.cpu().item() for loss in outer_loss_list_fDRO]
    #outer_loss_linear = [ loss.cpu().item() for loss in outer_loss_list_linear]
   
    #print("train loss linear",outer_loss_linear)
    test_loss_DRO = [loss.cpu().item() for loss in test_loss_list]
    #test_loss_DRO_1 = [loss.cpu().item() for loss in test_loss_list_1]
    #test_loss_DRO_2 = [loss.cpu().item() for loss in test_loss_list_2]
    
    test_loss_base_DRO = [loss.cpu().item() for loss in test_loss_list_base_DRO]
   
    test_loss_fDRO = [ loss.cpu().item() for loss in test_loss_list_fDRO]
    #test_loss_fDRO_1 = [ loss.cpu().item() for loss in test_loss_list_fDRO_1]
    #test_loss_fDRO_2 = [ loss.cpu().item() for loss in test_loss_list_fDRO_2]
   
    test_loss_linear = [ loss.cpu().item() for loss in test_loss_list_linear]
    #test_loss_linear_1 = [ loss.cpu().item() for loss in test_loss_list_linear_1]
    #test_loss_linear_2 = [ loss.cpu().item() for loss in test_loss_list_linear_2]
    #print("test loss is", test_loss_DRO)
    #print("test loss is", test_loss_DRO_1)
    #print("test loss is", test_loss_DRO_2)
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
    #plt.legend()
    #plt.show()
    #fig.savefig(f'train_logistic_{attack_method}_{epsilon}.png')

    fig=plt.figure()
    plt.loglog(epochs_list, test_loss_DRO,'-o',color='red',label="SDRO2"+f" $\epsilon={epsilon}$")
    #plt.loglog(epochs_list, test_loss_DRO_1, '-o',color = 'red',label="SDRO " + r"p=0.1")
    #plt.loglog(epochs_list, test_loss_DRO_2,'-o', color = 'red',label="SDRO " + r"p=0.5")
    plt.loglog(epochs_list, test_loss_base_DRO,'-+', color = 'orange',label = "SDRO1 "+ f" $\epsilon={epsilon}$")
    plt.loglog(epochs_list, test_loss_fDRO,'-*',color = 'green', label = "fDRO "+ f" $\epsilon={epsilon}$")
    #plt.loglog(epochs_list, test_loss_fDRO_1,'-*', color = 'green',label = "fDRO "+ r"p=0.1")
    #plt.loglog(epochs_list, test_loss_fDRO_2,'-*', color = 'green',label = "fDRO "+ r"p=0.5")
    plt.loglog(epochs_list, test_loss_linear,'-+', color = 'blue',label = "ERM "+ f" $\epsilon={epsilon}$")
    #plt.loglog(epochs_list, test_loss_linear_1,'-+', color = 'blue', label = "ERM "+ r"p=0.1")
    #plt.loglog(epochs_list, test_loss_linear_2,'-+', color = 'blue',label = "ERM "+ r"p=0.5")
    plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    plt.title("Test Loss Curve")
    plt.legend(loc = 'upper right')
    #plt.tight_layout()
    plt.savefig("your_plot.png")

    plt.show()
    fig.savefig(f'test_logistic_{attack_method}_{epsilon}.png')

    """
    hyper-parameters starts here.
   
    """

    python_file = "logistic.py"

    # Output file
    output_file = "hyperparams_logistic_regression.txt"
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