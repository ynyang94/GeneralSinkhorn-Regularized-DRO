# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:52:14 2025

@author: ynyang94
"""

import os
import shutil

def reorganize_dataset(root, output_root):
    """
    Reorganize flat dataset structure into subfolders by class for ImageFolder or DatasetFolder.

    Args:
        root (str): Path to the dataset with all images in a single directory.
        output_root (str): Path to save the reorganized dataset.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for file in os.listdir(root):
        if file.endswith('.png'):  # Only process PNG files
            # Extract label from the filename (e.g., "0_image1.png")
            label = file.split('_')[0]  # Get the label (e.g., "0" from "0_image1.png")
            label_dir = os.path.join(output_root, label)

            # Create the label directory if it doesn't exist
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            # Move the file into the appropriate subfolder
            shutil.move(os.path.join(root, file), os.path.join(label_dir, file))

# Paths
original_root = "C:/Users/ynyang94/Documents/MNIST-M/MNIST-M/training/"  # Path to the flat dataset
reorganized_root = "C:/Users/ynyang94/Documents/MNIST-M/MNIST-M/training/"  # Path for the reorganized dataset

# Reorganize dataset
reorganize_dataset(original_root, reorganized_root)
print("Dataset reorganized successfully!")