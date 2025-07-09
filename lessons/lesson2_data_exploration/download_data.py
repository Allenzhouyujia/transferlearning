#!/usr/bin/env python3
"""
Flowers102 Dataset Download Script
This script downloads the Flowers102 dataset for use in subsequent lessons.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

def download_flowers102():
    """Download Flowers102 dataset"""
    print("Flowers102 Dataset Download")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    # Basic transform for downloading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print("\nDownloading Flowers102 dataset...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        # Download training set
        print("\n1. Downloading training set...")
        trainset = torchvision.datasets.Flowers102(
            root=data_dir,
            split='train',
            download=True,
            transform=transform
        )
        print(f"   Training set: {len(trainset)} samples")
        
        # Download validation set  
        print("\n2. Downloading validation set...")
        validset = torchvision.datasets.Flowers102(
            root=data_dir,
            split='val',
            download=True,
            transform=transform
        )
        print(f"   Validation set: {len(validset)} samples")
        
        # Download test set
        print("\n3. Downloading test set...")
        testset = torchvision.datasets.Flowers102(
            root=data_dir,
            split='test',
            download=True,
            transform=transform
        )
        print(f"   Test set: {len(testset)} samples")
        
        print("\n" + "=" * 60)
        print("FLOWERS102 DATASET DOWNLOAD COMPLETED!")
        print("=" * 60)
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(trainset):,}")
        print(f"  Validation samples: {len(validset):,}")
        print(f"  Test samples: {len(testset):,}")
        print(f"  Total samples: {len(trainset) + len(validset) + len(testset):,}")
        print(f"  Number of classes: 102")
        print(f"  Image size: 224x224 pixels")
        print(f"  Data location: {os.path.abspath(data_dir)}")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nYou can now run the main lesson2_data_exploration.ipynb notebook!")
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = download_flowers102()
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload failed. Please try again.") 