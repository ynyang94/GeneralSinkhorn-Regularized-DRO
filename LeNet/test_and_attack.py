# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:43:47 2025

@author: ynyang94
"""

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

def generate_pgd_attacked_dataset(model, dataset, checkpoint_dir ,epoch,epsilon, iters, batch_size,train_method ):
    """
    Generate a dataset with PGD-attacked adversarial examples.

    Args:
        model: The trained model to attack.
        dataset: The dataset to attack (e.g., test dataset).
        epsilon: Maximum perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of attack iterations.
        batch_size: Batch size for DataLoader.

    Returns:
        adv_dataset: A TensorDataset containing the adversarial examples and their original labels.
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    all_adv_images = []
    all_labels = []

    for _, images, labels in dataloader:
        images, labels = images.clone(), labels.clone()  # Move to GPU if available
        
        
        # Generate adversarial examples using PGD
        # change here for different norms of PGD attack
        adv_images = l2_pgd_attack(model, images, labels, checkpoint_dir,epsilon, iters, epoch, train_method)
        model.eval()  # Set the model to evaluation mode
        #adv_images = sign_gradient_attack(model, images, labels, epsilon)
        # Store adversarial examples and labels
        all_adv_images.append(adv_images.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Concatenate all batches into a single dataset
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create a TensorDataset with the adversarial examples and labels
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    return adv_dataset
def generate_mim_attacked_dataset(model, dataset, checkpoint_dir,epoch,epsilon,mu, iters, batch_size,train_method ):
    """
    Generate a dataset with PGD-attacked adversarial examples.

    Args:
        model: The trained model to attack.
        dataset: The dataset to attack (e.g., test dataset).
        epsilon: Maximum perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of attack iterations.
        batch_size: Batch size for DataLoader.

    Returns:
        adv_dataset: A TensorDataset containing the adversarial examples and their original labels.
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    all_adv_images = []
    all_labels = []

    for _, images, labels in dataloader:
        images, labels = images.clone(), labels.clone()  # Move to GPU if available
        
        
        # Generate adversarial examples using PGD
        # change here for different norms of PGD attack
        adv_images = momentum_iterative_attack(model, images, labels, checkpoint_dir ,epsilon, iters, mu, epoch, train_method)
        model.eval()  # Set the model to evaluation mode
        #adv_images = sign_gradient_attack(model, images, labels, epsilon)
        # Store adversarial examples and labels
        all_adv_images.append(adv_images.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Concatenate all batches into a single dataset
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create a TensorDataset with the adversarial examples and labels
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    return adv_dataset

def generate_fgsm_attacked_dataset(model, dataset,checkpoint_dir, epoch, epsilon, batch_size,train_method):
    """
    Generate a dataset with PGD-attacked adversarial examples.

    Args:
        model: The trained model to attack.
        dataset: The dataset to attack (e.g., test dataset).
        epsilon: Maximum perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of attack iterations.
        batch_size: Batch size for DataLoader.

    Returns:
        adv_dataset: A TensorDataset containing the adversarial examples and their original labels.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #model.eval()  # Set the model to evaluation mode

    all_adv_images = []
    all_labels = []

    for _, images, labels in dataloader:
        images, labels = images.clone(), labels.clone()  # Move to GPU if available
        

        # Generate adversarial examples using PGD
        # change here for different norms of PGD attack
        adv_images = sign_gradient_attack(model, images, labels, checkpoint_dir,epoch,epsilon, train_method)

        # Store adversarial examples and labels
        all_adv_images.append(adv_images.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Concatenate all batches into a single dataset
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create a TensorDataset with the adversarial examples and labels
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    return adv_dataset




def pgd_attack(model, images, labels, checkpoint_dir, epsilon,iters, epoch, train_method ):
    """
    Perform Projected Gradient Descent (PGD) attack.

    Args:
        model: The trained model to attack.
        images: The original images (input batch) of shape [B, C, H, W].
        labels: True labels corresponding to the images.
        epsilon: Maximum allowable perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of iterations for the attack.

    Returns:
        perturbed_images: Adversarial images generated using PGD attack.
    """
    # Clone the input images to create adversarial examples
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.eval()
    model.load_state_dict(state_dict)
   
    
    images = images.clone().detach()
    original_images = images.clone().detach()
    
    alpha = epsilon/10
   
    for _ in range(iters):
        # Forward pass through the model
        # Detach the tensor and re-enable gradient tracking
        
        
        images = images.requires_grad_()
        model.eval()
        outputs = model(images)
        loss = model.cross_entropy_metric(outputs, labels)
        
        if images.grad is not None:  # Clear existing gradients
            images.grad.zero_()

        # Backward pass to compute gradients
        loss.backward()

        # Perform PGD update: Gradient ascent
        with torch.no_grad():
            grad = images.grad
            # l_infinity attack
            adv_images = images +alpha * grad.sign()

            # Project back to the epsilon-ball and clip to [0, 1]
            eta = torch.clamp(adv_images-original_images, min=-epsilon, max = epsilon)
            #print('eta', eta.abs().max())
            images = torch.clamp(original_images+eta, min = -1.0, max = 1.0)
        #delta = (images - original_images).abs().max()
        #print("max |delta| =", delta.item())          # should print 0.0 exactly
        

    return images.detach()

def l2_pgd_attack(model, images, labels, checkpoint_dir ,epsilon, iters, epoch , train_method ):
    """
    Perform Projected Gradient Descent (PGD) attack.

    Args:
        model: The trained model to attack.
        images: The original images (input batch) of shape [B, C, H, W].
        labels: True labels corresponding to the images.
        epsilon: Maximum allowable perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of iterations for the attack.

    Returns:
        perturbed_images: Adversarial images generated using PGD attack.
    """
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    

    # Clone the input images to create adversarial examples
    original_images = images.clone().detach()
    delta = torch.randn_like(original_images)                           # (B,C,H,W)

    # -- L2 norm of each perturbation --
    delta_norm = torch.norm(delta.view(delta.size(0), -1),     # (B,1)
                        p=2, dim=1, keepdim=True)

    # make it broadcastable: (B,1,1,1)
    delta_norm = delta_norm.view(-1, 1, 1, 1)                  # or .unsqueeze(2).unsqueeze(3)

    # random radius in [0,1] with same broadcastable shape
    rand_r = torch.rand_like(delta_norm)                       # (B,1,1,1)

    # scale to lie inside the L2 ball of radius
    delta = delta / (delta_norm + 1e-9) * rand_r * epsilon

    images = torch.clamp(original_images + delta, -1.0, 1.0)   # random start

    
    #original_images.requires_grad = True  # Enable gradient tracking
    alpha = epsilon/10
    for _ in range(iters):
        # Forward pass through the model
        images = images.detach().requires_grad_()
        model.eval()
        outputs = model(images)
        loss = model.cross_entropy_metric(outputs, labels)
        
        if images.grad is not None:  # Clear existing gradients
            images.grad.zero_()

        # Backward pass to compute gradients
        loss.backward()

        # Perform PGD update: Gradient ascent
        with torch.no_grad():
            grad = images.grad
            
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            grad = grad / (grad_norm)  # Add small value to prevent division by zero
            adv_images = images + alpha * grad
            
            eta = adv_images-original_images
            eta_norm = torch.norm(eta.view(eta.size(0), -1), p=2, dim=1, keepdim=True)

            # Project eta onto the L2 ball of radius epsilon
            scaling_factor = torch.min(torch.ones_like(eta_norm), epsilon / (eta_norm))
            eta = eta * scaling_factor.view(-1, 1, 1, 1)
            images = torch.clamp(original_images + eta, min=-1.0, max=1.0)

            # Project back to the epsilon-ball and clip to [0, 1]
            
            #images = torch.clamp(original_images+eta, min = 0, max = 1)

        # Detach the tensor and re-enable gradient tracking
        

    return images.detach()

def l1_pgd_attack(model, images, labels, checkpoint_dir ,epsilon, iters, epoch, train_method):
    """
    Perform Projected Gradient Descent (PGD) attack.

    Args:
        model: The trained model to attack.
        images: The original images (input batch) of shape [B, C, H, W].
        labels: True labels corresponding to the images.
        epsilon: Maximum allowable perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of iterations for the attack.

    Returns:
        perturbed_images: Adversarial images generated using PGD attack.
    """
    # Clone the input images to create adversarial examples
    original_images = images.clone().detach()
    images = images.clone().detach()
    #original_images.requires_grad = True  # Enable gradient tracking
    alpha = epsilon/10
    for _ in range(iters):
        images = images.detach().requires_grad_()
        # Forward pass through the model
        outputs = model(images)
        loss = model.cross_entropy_metric(outputs, labels)
        
        if images.grad is not None:  # Clear existing gradients
            images.grad.zero_()

        # Backward pass to compute gradients
        loss.backward()

        # Perform PGD update: Gradient ascent
        with torch.no_grad():
            grad = images.grad

            # Compute \ell_1 norm of gradients
            grad_abs = torch.abs(grad.view(grad.size(0), -1))
            grad_sign = grad.sign()
            grad_sum = torch.sum(grad_abs, dim=1, keepdim=True)
            grad = grad_sign * (grad_abs / (grad_sum + 1e-8)).view_as(grad)  # Normalize gradient

            # PGD step
            adv_images = images + alpha * grad

            # Compute \ell_1 norm of perturbation
            eta = adv_images - original_images
            eta_abs = torch.abs(eta.view(eta.size(0), -1))
            eta_sum = torch.sum(eta_abs, dim=1, keepdim=True)
            eta = eta / (eta_sum.view(-1, 1, 1, 1) + 1e-8) * torch.min(torch.ones_like(eta_sum), epsilon / eta_sum).view(-1, 1, 1, 1)

            # Final adversarial images
            images = torch.clamp(original_images + eta, min=-1.0, max=1.0)

            # Detach the tensor and re-enable gradient tracking
        

    return images.detach()


def momentum_iterative_attack(model, images, labels, checkpoint_dir ,epsilon, iters, mu, epoch, train_method):
    """
    Generate adversarial examples using the Momentum Iterative Method (MIM).
    
    Parameters:
        model (torch.nn.Module): The target model.
        images (torch.Tensor): Original input images (batch_size, C, H, W).
        labels (torch.Tensor): True labels for the images.
        epsilon (float): Maximum perturbation allowed (L_infinity norm).
        alpha (float): Step size for each iteration.
        iters (int): Number of attack iterations.
        mu (float): Momentum term (default is 1.0).
    
    Returns:
        adv_images (torch.Tensor): Adversarial examples.
    """
    # Ensure the model is in evaluation mode
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Clone original images for modification
    adv_images = images.clone().detach().requires_grad_(True)
    
    # Initialize the gradient momentum
    grad_momentum = torch.zeros_like(images).detach()
    alpha = 2*epsilon/iters
    for i in range(iters):
        # Forward pass and compute loss
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Compute gradients
        loss.backward()
        grad = adv_images.grad.data
        
        # Normalize the gradients (L1 normalization)
        grad_norm = grad.abs().sum(dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-8)
        
        # Update gradient momentum
        grad_momentum = mu * grad_momentum + grad
        
        # Update adversarial images with momentum
        adv_images = adv_images + alpha * grad_momentum.sign()
        
        # Project back to the L_infinity ball
        perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + perturbation, min=0, max=1).detach()
        
        # Detach and enable gradient for the next iteration
        adv_images.requires_grad_(True)
    
    return adv_images



def sign_gradient_attack(model, inputs, labels, checkpoint_dir,epoch,epsilon, train_method):
    """
    Perform Sign Gradient Attack.
    Args:
        model: The model to attack.
        inputs: Original inputs.
        labels: True labels.
        epsilon: Perturbation strength.
    Returns:
        Perturbed inputs.
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    labels = labels.clone().detach()

    # Set model to evaluation mode
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
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


def test_or_validation(model, test_loader, checkpoint_dir, num_epochs, interval, train_method, device = 'cpu'):
    #global device
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
                for data, label in test_loader:
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
        print(f"classification accuracy up to {epoch} is {acc_list}")
    return test_loss_list
