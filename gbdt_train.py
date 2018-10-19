#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
import xgboost as xgb
import datetime
from sklearn import preprocessing

def getweekday(x):
    '''
    :param x: YYMMDD
    :return: weekday
    '''
    sdt = str(x)
    year = int(sdt[0:2])
    month = int(sdt[2:4])
    day = int(sdt[4:6])
    dt = datetime.date(year, month, day)
    weekday = dt.weekday()
    return weekday

datapath = "."
trainfile = os.path.join(datapath ,"train_sample2.csv")
df = pd.read_csv(trainfile,dtype={"C15":str,"C16":str})
df["size"] = df["C15"].str.cat(df["C16"], sep="_")
# 将hour列拆分为
df["hour1"] = df["hour"].map(lambda x: str(x)[6:8])
df["day"] = df["hour"].map(lambda x: str(x)[4:6])
df["weekday"] = df["hour"].map(lambda x: getweekday(x))
df = df.drop(["id", "hour", "C15", "C16"], axis=1)

# specify parameters via map
param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}
num_round = 4
y_train = df["click"]
df = df.drop(["click"],axis=1)
#print(df.head())
df["C1"] = df["C1"].astype("category")
print("C1")
print(df["C1"])

print(y_train)
le = preprocessing.LabelEncoder()

X_train = le.fit_transform(df.values)
print(X_train.head())
dtrain = xgb.DMatrix(X_train,label = y_train)

bst = xgb.train(param, dtrain, num_round)
