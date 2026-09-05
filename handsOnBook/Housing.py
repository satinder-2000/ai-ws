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

print("\nremove the income_cat attribute\n")
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
    
print("\vDiscover and Visulizae the Data to Gain Insights\n")
housing = strat_train_set.copy()

housing.plot(kind="scatter", x ="longitude", y="latitude")

housing.plot(kind="scatter", x ="longitude", y="latitude",alpha=0.1)

print("\nPlotting the house prices\n")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label ="population", figsize=(10,7),
             c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

print("\n-----Looking for Correlations-----\n")
#corr_matrix = housing.corr()
#corr_matrix = housing.corr()
#corr_matrix = housing.corr()
#corr_matrix["median_house_value"].sort_values(ascending=False)
print("\nUsing pandas scatter_matrix() to check the correlations\n")
from pandas.plotting import scatter_matrix

attributes=["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

print("\nZoom into the correlation between mediand_income and median_house_value\n")
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
print("\nSome interesting attribute combinations\n")
housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
print("\n housing['rooms_per_household'] \n",housing["rooms_per_household"])
housing["bedrooms_per_room"]= housing["total_bedrooms"]/housing["total_rooms"]
print("\n housing['bedrooms_per_room'] \n",housing["bedrooms_per_room"])
housing["population_per_household"]= housing["population"]/housing["households"]
print("\n housing['population_per_household'] \n",housing["population_per_household"])

print("\n take a look at correlation matrix again\n")
#corr_matrix=housing.corr()
#corr_matrix["median_house_value"].sort_values(ascending=False)
print("\n-----Prepare the Data for ML Algorithms-----\n")
print("\nSeperate the predictors and the labels\n")
housing=strat_train_set.drop("median_house_value", axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

print("\n-----Data Cleaning-----\n")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
#median can only be calculated on numeric values so drop ocean_proximity
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
print("\nimputer.statistics_\n",imputer.statistics_)
print("\nhousing_num.median().values\n",housing_num.median().values)
print("\nUsing the 'trained' imputer to transforrm the training set")
print("by replacing missing values with the learned medians")
X = imputer.transform(housing_num)
#print("\n imputer.transform(housing_num)\n", X)
print("\nConvert th Numpy array above into a pandas DataFrame\n")
housing_tr=pd.DataFrame(X, columns=housing_num.columns,
                        index=housing_num.index)
print("\nhousing_tr:\n",housing_tr)
print("\n-----Handling Text and Categorical Attributes-----\n")
housing_cat=housing[["ocean_proximity"]]
print("\nhousing_cat.head(10):\n",housing_cat.head(10))
print("\n-----Convert the attrinutes to numeric values as most of ML algorithms prefer to work with numbers-----\n")
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
print("\nhousing_cat_encoded[:10]\n",housing_cat_encoded[:10])
print("\nordinal_encoder.categories_\n",ordinal_encoder.categories_)

print("\n-----Applying one-hot encoding-----\n")
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("\nhousing_cat_1hot.toarray():\n",housing_cat_1hot.toarray())