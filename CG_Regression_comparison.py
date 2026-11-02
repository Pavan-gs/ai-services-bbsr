# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:45:11 2018

@author: Deepstrats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("E:/data/Comparison.csv")

plt.scatter(y='Profit',x='M.cost',data=dataset)

x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
model1 = lm.fit(x,y)

lm.predict([[6.5]])

plt.scatter(x,y,color="blue")
plt.plot(x,model1.predict(x),color="red")

from sklearn.preprocessing import PolynomialFeatures

pol = PolynomialFeatures(degree=3)

x_po = pol.fit_transform(x)

pol.fit(x_po,y)

pol1 = LinearRegression()

pol1.fit(x_po,y)

pol1.predict(pol.fit_transform([[6.5]]))

plt.scatter(x,y,color="blue")
plt.plot(pol1.predict(pol.fit_transform(x)), color="red")

#SVM

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')

regressor.fit(x,y)

pred_svm = regressor.predict([[8.5]])
pred_svm

plt.scatter(x,y,color="blue")
plt.plot(x,regressor.predict(x),color="red")

#Decision Tree

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=5)

regressor.fit(x,y)

regressor.predict([[6.5]])

plt.scatter(x,y,color="blue")
plt.plot(x,regressor.predict(x),color="red")

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="blue")
plt.plot(x_grid, regressor.predict(x_grid),color="red")

#Randomforest

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=5)

regressor.fit(x,y)

regressor.predict([[6.5]])





