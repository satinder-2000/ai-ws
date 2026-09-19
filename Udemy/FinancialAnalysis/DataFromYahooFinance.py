#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:09:55 2026

@author: singh
"""
import yfinance as yf
import pandas as pd

df_yahoo = yf.download("AAPL",
                       start='2010-01-01',
                       end='2010-12-31',
                       progress=False)

print(df_yahoo.head())