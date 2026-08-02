#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:35:44 2026

@author: singh
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DataPrefetcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using devices: {device}") 

# Set a fixed seed for reproducibility
torch.manual_seed(42)

print("-----Creating a Sample Dataset------")
print()

class SyntheticDataSet(Dataset):
    """A synthetic dataset that simulates expensive data transformations."""
    
    def __init__(self, size=10000, feature_dim=224, transform_delay=0.001):
        self.size=size
        self.feature_dim=feature_dim
        self.transform_delay= transform_delay
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
       # Generate data lazily to avoid pre-allocating large tensors
       data = torch.randn(3, self.feature_dim, self.feature_dim)
       label = torch.randint(0, 10. (1,)).item()
       if self.transform_delay > 0:
           time.sleep(self.transform_delay)
       return data, label


class SyntheticDataSetBatched(Dataset):
    """Same as SyntheticDataset but with __getitems__ for batched fetching."""
    
    def __init__(self, size=10000, feature_dim=224, transform_delay=0.001):
        self.size=size
        self.feature_dim=feature_dim
        self.transform_delay= transform_delay
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
       # Generate data lazily to avoid pre-allocating large tensors
       data = torch.randn(3, self.feature_dim, self.feature_dim)
       label = torch.randint(0, 10. (1,)).item()
       if self.transform_delay > 0:
           time.sleep(self.transform_delay)
       return data, label
   
    def __getitems__(self, indices):
       """Fetch multiple items at once — enables vectorized generation.

        Instead of N individual __getitem__ calls (each with its own
        overhead), this generates the entire batch in one shot using
        vectorized tensor operations.
       """
       n = len(indices)
       # Vectorized generation: one call instead of N individual ones
       data = torch.randn(3, self.feature_dim, self.feature_dim)
       labels = torch.randint(0, 10. (n,))
       # Simulate batch-level I/O: one sleep for the whole batch,
       # not one per sample (e.g., one DB query for N rows)
       if self.transform_delay > 0:
           time.sleep(self.transform_delay)
       return [(data[i], labels[i].item()) for i in range(n)]
    
print("-----Shared Training Infrastructure------")
print()

benchmark_dataset = SyntheticDataSet(size=512, feature_dim=224, transform_delay=0.005)

class SmallTransformerModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_side = 7, strider = 2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_side = 3, strider = 2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,  batch_first=True
        )
        self.tranformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.features(x) #(B, 64, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # (B, 49, 64)
        x = self.tranformer(x) # (B, 49, 64)
        x = x.mean(dim=1) #(B, 64)
        return self.classifier(x)
    

def create_model():
    """Create a conv+transformer model for benchmarking."""
    return SmallTransformerModel().to(device)

def train_and_benchmark(loader, max_batches=160, epochs=10, prefetch_device=None):
    """Train a model over multiple epochs and return elapsed time and average loss.

    Running multiple epochs (10) with a small dataset ensures many epoch
    boundaries, making persistent_workers' startup savings visible.

    Args:
        loader: A DataLoader to iterate over.
        max_batches: Maximum total number of batches to process across all epochs.
        epochs: Number of epochs to iterate (re-iterates the loader each epoch).
        prefetch_device: If set, wraps the loader in a DataPrefetcher each epoch
            for overlapping H2D transfers. Data arrives already on device.

    Returns:
        Tuple of (elapsed_seconds, average_loss).
    """
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        if prefetch_device is not None:
            data_iter = DataPrefetcher(loader, prefetch_device)
        else:
            data_iter = loader
            
        for data, labels in data_iter:
            if prefetch_device is None:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
            output = model(data)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backword()
            optimizer.step()
            
    