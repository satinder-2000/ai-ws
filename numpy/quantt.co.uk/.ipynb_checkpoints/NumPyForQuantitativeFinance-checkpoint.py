# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
print("\n-----Arrays, Not Lists-----\n")
# Simulate a year of daily returns
np.random.seed(42)
returns = np.random.normal(0.0005, 0.02, 252)

# Basic statistics — no loops needed
mean_return = returns.mean()
daily_vol = returns.std()
annual_vol = daily_vol * np.sqrt(252)
sharpe = (mean_return * 252) / annual_vol

print(f"Annualised return: {mean_return * 252:.2%}")
print(f"Annualised volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")


print("\n-----Vectorisation: The Core Concept-----")
print("Vectorisation means applying an operation to an entire array at once\n")

print("Slow: Python loop (~150ms for 1M elements)")
prices_list = list(range(1_000_000))
results = [p * 1.02 for p in prices_list]

print("\nFast: vectorised NumPy (~2ms for 1M elements)")
prices_arr = np.arange(1_000_000, dtype=np.float64())
results = prices_arr * 1.02

print("---Broadcasting---\n")
# Normalise each stock's returns by subtracting its mean
# returns_matrix shape: (252, 5) — 252 days, 5 stocks
returns_matrix = np.random.normal(0.001, 0.02,(252,5))
print("returns_matrix.head():\n",returns_matrix[0])
# means shape: (5,) — one mean per stock
means = returns_matrix.mean(axis=0)
print("\nmeans[0]: ",means[0])

# Broadcasting subtracts each column's mean automatically
demeaned = returns_matrix - means # Shape: (252, 5)
print("demeaned.head():\n",demeaned[0])