# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:22:46 2025

@author: ynyang94
"""

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        """
        Args:
            dataset: A tensor where features are all columns except the last, 
                     and labels are in the last column.
            transform: Optional; a function/transform to apply to the data.
            target_transform: Optional; a function/transform to apply to the label.
        """
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract data and label
        data = self.dataset[idx, :-1]  # All columns except the last
        label = self.dataset[idx, -1]  # Last column as label
        # Apply transformations
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        # Return index, data, and label
        return idx, data, label