#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 05:23:36 2026

@author: singh
"""
import os
import numpy as np
import pandas as pd

HOUSING_PATH = os.path.join('/home/singh/temp/misc')

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing = load_housing_data()

print("\n-----Custom Transformers-----\n")
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kwargs
        self.add_bedrooms_per_room=add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self #nothing to do
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:,households_ix]
        poplation_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, poplation_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, poplation_per_household]
        
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
            
print("\n-----Feature Scaling-----\n")