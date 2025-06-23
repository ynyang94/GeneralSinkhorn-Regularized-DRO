import SinkhornDRO
import DataGenerator
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
import copy
import os
# import cvxpy as cp
import pdb
import math
import re
torch.manual_seed(42)



def sinkhorn_train(model, lr,eta0_list,train_dataloader, num_epochs, checkpoint_dir, interval=5):
    global device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.unsqueeze(1)
            num_zeta = batch_data.size(0)
            batch_data.to(device)
            batch_label.to(device)
            #print(batch_data.size(), batch_label.size())
            expected_loss = 0
            for i in range(num_zeta):
                # generate corrput data
                inputd,outputd = model.generate_corrput_data(batch_data[i,:], batch_label[i,:])
                inputd = inputd.to(device)
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

def sinkhorn_wang_train(model, lr,train_dataloader, num_epochs, checkpoint_dir, interval=5):
    global device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.unsqueeze(1)
            num_zeta = batch_data.size(0)
            batch_data.to(device)
            batch_label.to(device)
            #print(batch_data.size(), batch_label.size())
            expected_loss = 0
            for i in range(num_zeta):
                # generate corrput data
                inputd,outputd = model.generate_corrput_data(batch_data[i,:], batch_label[i,])
                inputd = inputd.to(device)
                
                #outputd = outputd.squeeze(dim=1)
                outputd = outputd.to(device)
                #print(inputd.size(), outputd.size())
                #predictions = torch.empty((num_xi,1))
                # generate predictions.
                #for j in range(num_xi):
                predictions = model(inputd)
                #print(model.state_dict())
                
                predictions = predictions.to(device)
                #print(predictions.size())
                inner_loss = model.baseline_sinkDRO(predictions, batch_data[i,:], inputd, outputd)
                #inner_grad=model.gradient_eta(predictions, eta, X_train[i,:], y_train[i,:])
                expected_loss += inner_loss
            expected_loss = (model.regularization*model.epsilon)*(expected_loss/num_zeta)
            #print(outer_loss)
            optimizer.zero_grad()  # Clear gradients from the last step
            expected_loss.backward()
            #total_norm = 0.0
            #for p in model.parameters():
            #    if p.grad is not None:
            #        param_norm = p.grad.data.norm(2)  # L2 norm
            #        total_norm += param_norm.item() ** 2

            #total_norm = total_norm ** 0.5
            #print(f"Total gradient norm: {total_norm}")
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
    eta = torch.tensor(eta, requires_grad= True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outer_loss_list = []
    for epoch in range(interval,num_epochs+interval, interval):
        model.train()
        #epoch_outer_loss = 0
        expected_loss = 0
        for idx, batch_data, batch_label in train_dataloader:
            batch_label = batch_label.unsqueeze(1)
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
            train_label = train_label.unsqueeze(1)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            predictions = model(train_data)
            predictions = predictions.to(device)
            train_loss = model.mse_metric(predictions, train_label)
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
    

def test_or_validation(model, test_loader, checkpoint_dir, num_epochs, interval=5, train_method = "DRO"):
    global device
    test_loss_list = []
    with torch.no_grad():
        for epoch in range(interval, num_epochs + interval, interval):
            # Construct the checkpoint path based on the current epoch
            checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
        
            # Load the model's state dictionary from the checkpoint
            state_dict = torch.load(checkpoint_path, weights_only=True)  # Use weights_only=True here
            model.load_state_dict(state_dict)
            model.eval()  # Set model to evaluation mode

            expected_eval_loss = 0
            with torch.no_grad():
                for idx, data, label in test_loader:
                    label = label.unsqueeze(1)
                    data = data.to(device)
                    label = label.to(device)
                    #num_zeta = data.size(0)
                    predictions = model(data)
                    predictions = predictions.to(device)
                    test_loss = model.mse_metric(predictions, label)
                    expected_eval_loss += test_loss
                expected_eval_loss = expected_eval_loss/len(test_loader)
            test_loss_list.append(expected_eval_loss.detach())
        print(test_loss_list)
    return test_loss_list



if __name__ == "__main__":
    """
    Hyper-parameter setting for dataset 
    (You should NOT change synethic dataset unless it's too hard to converge.)
    """
    num_samples_train= 3000  #number of samples for training
    num_samples_test = 500
    input_dim=10   #dimensionality
    # mean, std for input data
    mean = 0
    std = 0.3
    # y_std is for label
    y_std= 0.5
    multi_mean = 0.5*torch.ones((input_dim,))
    covariance = 0.1*torch.eye(input_dim)
    SEED = 32
    BATCH_SIZE = 64
    EVAL_SIZE = 128
    num_epochs = 80
    # white-noise attack on test data
    #perturbation_strength_0 = 0.0
    perturbation_strength_1 = 0.15
    perturbation_strength_2 = 0.30
    
    interval = 8
    """
    ----- Data Generation
    """
    torch.manual_seed(SEED)
    train_dataset = DataGenerator.SyntheticDataset(num_samples_train, input_dim, mean, std, multi_mean, covariance, y_std, SEED)
    device = torch.device("cpu")
    #test_dataset_0 = DataGenerator.PerturbedDataset(num_samples_test, input_dim, mean, std, multi_mean, covariance, y_std, SEED, perturbation_strength_0)
    test_dataset_1 = DataGenerator.PerturbedDataset(num_samples_test, input_dim, mean, std, multi_mean, covariance, y_std, SEED, perturbation_strength_1)
    test_dataset_2 = DataGenerator.PerturbedDataset(num_samples_test, input_dim, mean, std, multi_mean, covariance, y_std, SEED, perturbation_strength_2)

    #X_train = dataset[:, :-1]
    #y_train = dataset[:, -1]
    #y_train = y_train.unsqueeze(1)
    #print(dataset.size())
    #print(X_train.size())
    #print(y_train.size())
    ## represents x^*
    """
    this one generates the ground-truth model parameter we want to learn
    """
    ground_truth_parameter = train_dataset.generate_parameter()
    # Sample nominal distribution


    #multi_normal_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)


    #X_train= multi_normal_dist.sample((m,))
    #print(X_train.size())

    #y_train = torch.matmul(X_train,ground_truth_parameter)+torch.normal(0,y_std,(m,1))
    #print(y_train.size())

    """
    ---- Model parameter Initilization
    """
    #n = input_dim
    #m = num_samples
    # Intialize primal and dual variable.
    base_parameter = ground_truth_parameter.reshape((input_dim,))
    x0 = base_parameter+torch.normal(0.05,0.1, size =(input_dim,))
    #mean_x0 = x0.mean()
    #std_x0 = x0.std()
    #x0 = (x0 - mean_x0)/std_x0
    x0 = x0.requires_grad_(True)
    #print(x0.shape)
    # in-context dual variable initialization
    eta0_initialize = torch.normal(5.0,1.5, size =(num_samples_train,))
    mean_eta0 = eta0_initialize.mean()
    #std_eta0 = eta0_initialize.std()
    #eta0_initialize = (eta0_initialize - mean_eta0)/std_eta0
    eta0_list = eta0_initialize.clone().detach()


    x0 = x0.to(device)
    x0 = x0.view(1,-1)
    eta0_list = eta0_list.to(device)
    #X_train = X_train.to(device)
    #y_train = y_train.to(device)

    """
    ---- Sinkhorn DRO and other baseline model Initilaization.
    """

    loss_type = 'regression'
    model_DRO = SinkhornDRO.SinkhornDRO(input_dim,1,x0.clone().detach())
    model_DRO.num_xi = 8
    num_xi = model_DRO.num_xi
    model_DRO.regularization = 0.8
    model_DRO.epsilon = 1.0
    model_DRO.inner_loop = 5
    model_DRO.lr1 = 5e-2
    model_DRO.lr2 = 8e-2
    model_DRO.to(device)
    #inputd,outputd = model.generate_corrput_data(X_train[0,:], y_train[0,:])
    #print(inputd.size())
    #print(outputd.size())
    """
    -- Sinkhorn baseline from Wang, Jie's work'
    """
    model_base_DRO = sinkhorn_base.WangSinkhorn(input_dim,1,x0.clone().detach())
    model_base_DRO.num_xi = 8
    #num_xi = model_DRO.num_xi
    model_base_DRO.loss_type = loss_type
    model_base_DRO.noise = 0.2
    model_base_DRO.regularization = 0.8
    model_base_DRO.epsilon = 1.0
    model_base_DRO.lr1 = 1e-3
    model_base_DRO.to(device)
    
    """
    f-DRO initialization
    """
    model_fDRO = fDRO.fDRO(input_dim, 1, x0.clone().detach())
    model_fDRO.lr1 = 5e-4
    model_fDRO.regularization = 0.8
    model_fDRO.to(device)
    eta_fDRO = 0.8
    
    """
    ERM initialization
    """
    model_ERM = LinearERM.LinearModel(input_dim, 1, x0.clone().detach())
    ERM_lr = 1e-3
    model_ERM.to(device)
    """
    trainining and test dataset setup
    """

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,num_workers = 8,shuffle=False, pin_memory=True)
    #eval_dataloader = DataLoader(train_dataset, batch_size=EVAL_SIZE,num_workers = 8 ,shuffle=False, pin_memory=True)
    #test_dataloader_0 = DataLoader(test_dataset_0, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_1 = DataLoader(test_dataset_1, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)
    test_dataloader_2 = DataLoader(test_dataset_2, batch_size=EVAL_SIZE, num_workers = 8,shuffle = False, pin_memory=True)

    # address need to be changed.
    checkpoint_dir = r"C:\Users\ynyang94\OneDrive - Texas A&M University\Documents\checkpoints"
    # only change ERM learning rate.
    """
    training pipeline and loss here.
    """
    #outer_loss_list_base_DRO = sinkhorn_wang_train(model_base_DRO,model_base_DRO.lr1,train_dataloader, num_epochs, checkpoint_dir, interval)
    #outer_loss_list = sinkhorn_train(model_DRO,model_DRO.lr1,eta0_list.clone(),train_dataloader, num_epochs, checkpoint_dir, interval)
    
    #outer_loss_list_fDRO = fDRO_train(model_fDRO, model_fDRO.lr1, train_dataloader, num_epochs, checkpoint_dir, eta_fDRO, interval)
    #outer_loss_list_linear = ERM_train(model_ERM,ERM_lr, train_dataloader, num_epochs, checkpoint_dir, interval)
    #print(outer_loss_list)
    

    """
    test for sinkhorn DRO
    """
    #test_loss_list = test_or_validation(model_DRO,test_dataloader_0, checkpoint_dir, num_epochs, interval)
    test_loss_list_1 = test_or_validation(model_DRO,test_dataloader_1, checkpoint_dir, num_epochs, interval)
    test_loss_list_2 = test_or_validation(model_DRO,test_dataloader_2, checkpoint_dir, num_epochs, interval)
    
    """
    test for sinkhorn DRO baseline
    """
    #test_loss_list_base_DRO = test_or_validation(model_base_DRO,test_dataloader_0, checkpoint_dir, num_epochs, interval, train_method = 'wangDRO')
    test_loss_list_base_DRO1 = test_or_validation(model_base_DRO,test_dataloader_1, checkpoint_dir, num_epochs, interval, train_method = 'wangDRO')
    test_loss_list_base_DRO2 = test_or_validation(model_base_DRO,test_dataloader_2, checkpoint_dir, num_epochs, interval, train_method = 'wangDRO')
    
    """
    test for f-DRO
    """
    #test_loss_list_fDRO = test_or_validation(model_fDRO,test_dataloader_0, checkpoint_dir, num_epochs,interval, train_method = "fDRO")
    test_loss_list_fDRO_1 =  test_or_validation(model_fDRO,test_dataloader_1, checkpoint_dir, num_epochs,interval,train_method = "fDRO")
    test_loss_list_fDRO_2 =  test_or_validation(model_fDRO,test_dataloader_2, checkpoint_dir, num_epochs, interval,train_method = "fDRO")
    """
    test for linear ERM
    """
    #test_loss_list_linear = test_or_validation(model_ERM,
    #                                       test_dataloader_0, checkpoint_dir, num_epochs, interval, train_method = "Linear")
    test_loss_list_linear_1 = test_or_validation(model_ERM,
                                           test_dataloader_1, checkpoint_dir, num_epochs, interval, train_method = "Linear")
    test_loss_list_linear_2 = test_or_validation(model_ERM,
                                           test_dataloader_2, checkpoint_dir, num_epochs, interval, train_method = "Linear")
    
    #outer_loss_DRO = [loss.cpu().item() for loss in outer_loss_list]
    #outer_loss_fDRO = [loss.cpu().item() for loss in outer_loss_list_fDRO]
    #outer_loss_linear = [ loss.cpu().item() for loss in outer_loss_list_linear]
    
    #print("train loss linear",outer_loss_linear)
    #test_loss_DRO = [loss.cpu().item() for loss in test_loss_list]
    test_loss_DRO_1 = [loss.cpu().item() for loss in test_loss_list_1]
    test_loss_DRO_2 = [loss.cpu().item() for loss in test_loss_list_2]
    
    #test_loss_base_DRO = [loss.cpu().item() for loss in test_loss_list_base_DRO]
    test_loss_base_DRO_1 = [loss.cpu().item() for loss in test_loss_list_base_DRO1]
    test_loss_base_DRO_2 = [loss.cpu().item() for loss in test_loss_list_base_DRO2]
    
    #test_loss_fDRO = [ loss.cpu().item() for loss in test_loss_list_fDRO]
    test_loss_fDRO_1 = [ loss.cpu().item() for loss in test_loss_list_fDRO_1]
    test_loss_fDRO_2 = [ loss.cpu().item() for loss in test_loss_list_fDRO_2]
    
    #test_loss_linear = [ loss.cpu().item() for loss in test_loss_list_linear]
    test_loss_linear_1 = [ loss.cpu().item() for loss in test_loss_list_linear_1]
    test_loss_linear_2 = [ loss.cpu().item() for loss in test_loss_list_linear_2]
    #print("test loss is", test_loss_DRO)
    #print("test loss is", test_loss_DRO_1)
    #print("test loss is", test_loss_DRO_2)
    """
    plot starts here
    """
    epochs_list = list(range(interval, num_epochs + interval, interval))

    #fig=plt.figure()
    #plt.loglog(epochs_list, outer_loss_DRO,'-o',color = 'red', label="Train Loss by SDRO")
    #plt.loglog(epochs_list, outer_loss_linear,'-+',color = 'blue', label = "Mean Squard Loss by ERM ")
    #plt.loglog(epochs_list, outer_loss_fDRO,'-*',color = 'green', label = "Mean Squard Loss by fDRO ")

    #plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    #plt.title("Train Loss Curve")
    #plt.legend()
    #plt.show()
    #fig.savefig('train.png')

    fig=plt.figure()
    #plt.loglog(epochs_list, test_loss_DRO,'-o',color='red',label="SDRO2 "+r"p=0.0")
    plt.loglog(epochs_list, test_loss_DRO_1, '-*',color = 'red',label="SDRO2 " + f"p={perturbation_strength_1}")
    plt.loglog(epochs_list, test_loss_DRO_2,'-+', color = 'red',label="SDRO2 " + f"p={perturbation_strength_2}")
    
    #plt.loglog(epochs_list, test_loss_base_DRO,'-o',color='orange',label="SDRO1 "+r"p=0.0")
    plt.loglog(epochs_list, test_loss_base_DRO_1, '-*',color = 'orange',label="SDRO1 " + f"p={perturbation_strength_1}")
    plt.loglog(epochs_list, test_loss_base_DRO_2,'-+', color = 'orange',label="SDRO1 " + f"p={perturbation_strength_2}")
    
    #plt.loglog(epochs_list, test_loss_fDRO,'-o',color = 'green', label = "fDRO "+ r"p=0.0")
    plt.loglog(epochs_list, test_loss_fDRO_1,'-*', color = 'green',label = "fDRO "+ f"p={perturbation_strength_1}")
    plt.loglog(epochs_list, test_loss_fDRO_2,'-+', color = 'green',label = "fDRO "+ f"p={perturbation_strength_2}")
    
    #plt.loglog(epochs_list, test_loss_linear,'-o', color = 'blue',label = "ERM "+ r"p=0.0")
    plt.loglog(epochs_list, test_loss_linear_1,'-*', color = 'blue', label = "ERM "+ f"p={perturbation_strength_1}")
    plt.loglog(epochs_list, test_loss_linear_2,'-+', color = 'blue',label = "ERM "+ f"p={perturbation_strength_2}")
    plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    plt.title("Test Loss Curve")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("your_plot.png")

    plt.show()
    fig.savefig('test_laplacian_attack.png')

    """
    hyper-parameters starts here.
    
    """

    python_file = "Regression.py"

    # Output file
    output_file = "hyperparams_linear_regression.txt"
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