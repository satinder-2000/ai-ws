#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:15:10 2026

@author: singh
"""
print("https://github.com/satinder-2000/ai-ws.git")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors


print("\n-----load the data-----\n")

oced_bli = pd.read_csv('/home/singh/ai-ws/handsOnBook/datadir/oecd_bli_2015.csv', thousands=',')
gdp_per_capita= pd.read_csv('/home/singh/ai-ws/handsOnBook/datadir/gdp_per_capita.csv', thousands=','
                            ,delimiter='\t',
                            encoding='latin1', na_values="n/a")


print("\n-----Prepare the data-----\n")

#Source of the fn below:https://www.cnblogs.com/yaoz/p/6858417.html 
def prepare_country_stats(oecd_bli, gdp_per_capita):
    #get the pandas dataframe of GDP per capita and Life satisfaction
    #oecd_bli = oecd_bli["INEQUALITY" =="TOT"]
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    #gdp_per_capita.rename(columns=("2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country",inplace=True)
    #full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    return full_country_stats[["GDP per capita", 'Life satisfaction']]


country_stats = prepare_country_stats(oced_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]] 

#Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")

#Select a liners model
model = sklearn.linear_model.LinearRegression()
#model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

#Train the model
#lin_reg_model.fit(X, y)
model.fit(X, y)


#plot Regression model
t0, t1 = model.intercept_[0], model.coef_[0][0]
#t0, t1 = model.intercept_[0], model.coef_[0][0]
X = np.linspace(0, 110000, 1000)
plt.plot(X, t0 + t1 * X, "k")
plt.show()

#Make a prediction for Cyprus
X_new = [[22587]] #Cyprus GDP per capita
#print(lin_reg_model.predict(X_new))
print(model.predict(X_new))