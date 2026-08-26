#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:32:42 2026

@author: singh
"""
import pandas as pd
import numpy as np

print()
print("-----Creating Series-----")
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series: \n", s)

print()
print("-----Creating DataFrame-----")
dates = pd.date_range("20130101", periods=6)
print("DataFrame:\n",dates)

print()
print("-----DataFrame contains dictionary of objects-----")
df2 = pd.DataFrame(
    {
     "A":1.0,
     "B": pd.Timestamp("20130102"),
     "C":pd.Series(1, index=list(range(4)), dtype="float32"),
     "D":np.array([3] *4, dtype="int32"),
     "E": pd.Categorical(["test", "train", "test", "train"]),
     "F": "f00"
     }
)
print(df2)
print(df2.dtypes)
