#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys,os
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
housing_data = datasets.load_boston()
print(housing_data)
X,y = shuffle(housing_data.data,housing_data.target,random_state=7)
num_training = int(len(X) *0.8)
X_train,y_train = X[:num_training],y[:num_training]
X_test,y_test = X[num_training:],y[num_training:]

dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,y_train)
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred_dt)
evs = explained_variance_score(y_test,y_pred_dt)
print("Decision Tree performance")
print("Mean squared error=",round(mse,2))
print("Explained variance score = ",round(evs,2))

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ab_regressor.fit(X_train,y_train)
y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred_ab)
evs = explained_variance_score(y_test,y_pred_ab)
print("AdaBoosting performance")
print("Mean squared error=",round(mse,2))
print("Explainded variance score=",round(evs,2))



