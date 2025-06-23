# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:58:20 2025

@author: ynyang94
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained ResNet50 normalization
])

# Load CIFAR-10 dataset
batch_size = 256
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained ResNet50
resnet = resnet50(pretrained=True)
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 250),  # Replace with 100 features
    nn.ReLU(inplace=True)  # Optional, adds non-linearity
)

#resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the FC layer

# Freeze ResNet50 weights (optional for feature extraction)
#for param in resnet_feature_extractor.parameters():
    #param.requires_grad = False
    
def extract_features(dataloader, model, device):
    model.to(device)
    model.eval()  # Set to evaluation mode
    features, labels = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # Extract features
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten features
            features.append(outputs.cpu())
            labels.append(targets)
    
    # Concatenate features and labels
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

extract_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_features, train_labels = extract_features(train_loader, resnet, extract_device)
test_features, test_labels = extract_features(test_loader, resnet, extract_device)
saving_path= "C:/Users/ynyang94/Documents/Github/SinkhornDRO"
# Save the features and labels to a file
torch.save(
    {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    },
    "cifar10_resnet_features.pth"
)

print("Features saved to cifar10_resnet_features.pth")

