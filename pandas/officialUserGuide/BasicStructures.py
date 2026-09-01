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
print()
print("-----DataFrame.head() and tail()-----")
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list("ABCD"))
print("df:\n",df)
print()
print("df.head():\n",df.head())
print()
print("df.tail(3):\n",df.tail(3))
print()
print("df.index:\n",df.index)
print()
print("df.columns:\n",df.columns)
print()
print("to numpy() > without index or columns")
print("df.to_numpy():\n",df.to_numpy())
print()
print("df2.to_numpy():\n",df2.to_numpy())
print()
print("df.describe():\n",df.describe())
print()
print("df.T (Transpose):\n",df.T)
print()
print("df.sort_index() by an axis:\n",df.sort_index(axis=1, ascending=False))
print()
print("df.sort_values(by='B') :\n",df.sort_values(by='B'))
print()
print("-----Selection-----")
print("optimized pandas data access is recommended")
print()
print("Getitem ([]) > df['A']:\n", df["A"])
print()
print("alternatively, df.A:\n", df.A)
print()
print("subset/rearranging:df[['B','A']]:\n", df[["B","A"]])
print()
print("-----DataFrame slicing-----")
print()
print("df[0:3]:\n",df[0:3])
print()
print("df['20130102':'20130104']:\n",df['20130102':'20130104'])
print()
print("-----Selection by label-----")
print()
print("df.loc[dates[0]]:\n",df.loc[dates[0]])
print()
print("df.loc[:, ['A','B']]:\n",df.loc[:, ['A','B']])
print()
print("-----For label slicing, both endpoints are included:-----")
print()
print("df.loc['20130102':'20130104',['A','B']]:\n",df.loc['20130102':'20130104',['A','B']])
print("-----Selecting a single row and column label returns a scalar:-----")
print()
print("df.loc[dates[0],'A']\n",df.loc[dates[0],'A'])
print("or faster")
print("df.at[dates[0],'A']\n",df.at[dates[0],'A'])
print("-----Selection by position-----")
print()
print("df.iloc[3]:\n",df.iloc[3])
print()
print("df.iloc[3:5, 0:2]:\n",df.iloc[3:5, 0:2])
print("-----Lists of integer position locations:-----")
print("df.iloc[[1,2,4],[0,2]]:\n",df.iloc[[1,2,4],[0,2]])
print()
print("-----For slicing rows explicitly:-----")
print("df.iloc[1:3, :]:\n",df.iloc[1:3, :])
print()
print("-----For slicing columns explicitly:-----")
print("df.iloc[:, 1:3]:\n",df.iloc[:, 1:3])
print()
print("-----For getting a value explicitly:-----")
print("df.iloc[1, 1]:\n",df.iloc[1, 1])
print()
print("-----For getting fast access to a scalar (equivalent to the prior method):-----")
print("df.iat[1, 1]:\n",df.iat[1, 1])
print()
print("-----Boolean indexing-----")
print()
print("df[df['A']>0]:\n",df[df['A']>0])
print("Selecting values from a DataFrame where a boolean condition is met:")
print("df[df > 0]:\n",df[df > 0])
print()
print("Using isin() method for filtering")
dfcp=df.copy()
dfcp['E']=['one', 'one', 'two', 'three', 'four', 'three']
print("dfcp:\n",dfcp)
print()
print("dfcp[dfcp['E'].isin(['two','four'])]:\n",dfcp[dfcp['E'].isin(['two','four'])])
print()
print("-----Setting-----")
print()
print("Setting a new column automatically aligns the data by the indexes:")
s1 = pd.Series(
    [1, 2, 3, 4, 5, 6],
    index=pd.date_range("20130102", periods=6)
    )
print("s1:\n",s1)
df['F']=s1
print("Setting values by label")
df.at[dates[0],'A'] = 0
print("df:\n",df)
print("Setting values by position")
df.iat[0,1] = 0
print("df.iat[0,1] = 0:\n",df)
print("Setting by assigning with a NumPy array:")
df.loc[:,'D'] = np.array([5]* len(df))
print("df.loc[:,'D']:\n",df)
print("\nA where operation with setting\n")
df2 = df.copy()
df2[df2 > 0] = -df2
print("df2[df2 > 0] = -df2:\n",df2)

print("\n-----Missing data-----\n")
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0] : dates[1], 'E'] = 1
print("df1:\n",df1)
print("\nDataFrame.dropna() drops any rows that have missing data:\n")
print("df1.dropna(how='any'):\n",df1.dropna(how="any"))

print("\nDataFrame.fillna() fills missing data:\n")
print("df1.fillna(value=5):\n",df1.fillna(value=5))

print("\nisna() gets the boolean mask where values are nan:\n")
print("df1:\n",df1)
print("pd.isna(df1):\n",pd.isna(df1))