# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:18:44 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2] # Independent 
y = dataset.iloc[:,2] # dependent 


#Fitting Random forest regressor to dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)

y_predict = regressor.predict(6.5)

# Visualize

# X.grid = np.arange(min(X), max(X), 0.01)

X_grid = np.arange(1, 10, 0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title("True vs predict")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
