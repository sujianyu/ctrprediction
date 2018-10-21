#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
import sys,os
import xgboost as xgb
import datetime
from  sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from xgboost import plot_importance
from matplotlib import pyplot as plt
# 计算分类正确率
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
         'nthread': 4, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}
num_round = 5
y_train = df["click"]
df = df.drop(["click"],axis=1)

le = preprocessing.LabelEncoder()
columns = ["C1","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","size"]
for columnname in columns:
    df[columnname]= le.fit_transform(df[columnname])
X_train = df.values
print(X_train)
dtrain = xgb.DMatrix(X_train,y_train)

bst = xgb.train(param, dtrain, num_round)
#pred_leaf=True,
train_preds = bst.predict(dtrain,pred_leaf=False)
print(train_preds.shape)


train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
# 显示重要特征
plot_importance(bst)
plt.show()
#得到新特征
x_train_feature = bst.predict(dtrain,pred_leaf=True)
new_feature = DataFrame(x_train_feature)
print(new_feature.shape)
print(new_feature.head())

#对新特征使用onehot编码
enc = OneHotEncoder()
new_feature_onehot = enc.fit_transform(new_feature).toarray()
print(new_feature_onehot.shape)

#进行LR预测
# 定义LR模型
lr = LogisticRegression()
lr.fit(new_feature_onehot,y_train)
y_pred_xgblr1_proba = lr.predict_proba(new_feature_onehot)
y_pred_xgblr = lr.predict(new_feature_onehot)
#print(y_pred_xgblr)
total = len(y_train)
correct = 0
for index in range(total):
    if y_train[index] == y_pred_xgblr[index]:
        correct += 1
#print(y_pred_xgblr1_proba)
print("XGB _+ LR :",total,correct,correct * 1.0 /total)
