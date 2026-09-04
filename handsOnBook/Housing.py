#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:32:58 2026

@author: singh
"""
import os
import tarfile
import urllib
import pandas as pd

HOUSING_PATH = os.path.join('/home/singh/temp/misc')

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing = load_housing_data()
print("\nhousing.head():\n",housing.head())
print("\nhousing['ocean_proximity'].value_counts():\n",housing['ocean_proximity'].value_counts())
print("\nhousing.describe():\n",housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

print("\n-----Create a Test Set-----\n")

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print("\nlen(train_set)\n",len(train_set))
print("\nlen(test_set)\n",len(test_set))

from zlib import crc32

def test_set_ckeck(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32

def split_train_test_by_id(data,test_ratio, id_column):
    ids=data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_ckeck(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

print("\nUse row index as ID\n")

housing_with_id = housing.reset_index()#adds an 'index' column
train_set, test = split_train_test_by_id(housing_with_id, 0.2, "index")

#latitude and longitutes of a district are best candidates for id
housing_with_id["id"]=housing["longitude"]*1000 + housing["latitude"]
train_set, test_set= split_train_test_by_id(housing_with_id,0.2, "id")

print("\n------Using Scikit-Learn-----\n")
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)

print("\nUse pd.cut() fn to create an income_category attribute\n")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])
housing["income_cat"].hist()

print("\nStratified sampling based on income category\n")
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state =42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
    print("\nstrat_test_set['income_cat'].value_counts() / len(strat_test_set):\n",strat_test_set['income_cat'].value_counts() / len(strat_test_set))
