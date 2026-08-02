#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:10:44 2026

@author: singh
"""
print("Source: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html")
print()

print("Tensors are similar to NumPy’s ndarrays, \nexcept that tensors can run on GPUs or other hardware accelerators. ")

import torch
import numpy as np

print("-----Initilizing a Tensor-----")
print()

#Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print("Directly from data\n",x_data)

#From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("From a NumPy array\n", x_np)

#From another tensor
x_ones=torch.ones_like(x_data)
print("From another tensor\n")
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

print("With random or constant values\n")
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


print("-----Attributes of a Tensor-----")
print()

tensor = torch.rand(3, 4)
print(f"Shape of a tensor: {tensor.shape}")
print(f"Datatype of a tensor: {tensor.dtype}")
print(f"Device of a tensor: {tensor.device}")

print("-----Operations on Tensors-----")
print()
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First columns: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)