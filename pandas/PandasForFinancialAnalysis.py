#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:30:04 2026

@author: singh
"""
import pandas as pd
import numpy as np

#downloaded from https://www.kaggle.com/datasets/just4jcgeorge/stock-prices-csv
#then archive>cookbook>data>stock_prices
df = pd.read_csv("/home/singh/temp/kaggle/stock_prices.csv", parse_dates=["date"], index_col="date")

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())

print("\n-----Calculating Returns-----\n")
# Simple (arithmetic) returns
simple_returns = df.pct_change()

# Log (geometric) returns — preferred for statistical analysis
log_returns = np.log(df / df.shift(1))

# Cumulative returns — useful for plotting equity curves
cumulative = (1 + simple_returns).cumprod()

# Year-to-date return for each stock
ytd_return = (df.iloc[-1] / df.iloc[0]) - 1
print(ytd_return)
