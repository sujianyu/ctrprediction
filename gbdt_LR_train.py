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
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
import joblib
from sklearn.preprocessing import scale
import gc
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

def create_feature(data):
    data["size"] = data["C15"].str.cat(data["C16"], sep="_")
    # 将hour列拆分为
    data["hour1"] = data["hour"].map(lambda x: str(x)[6:8])
    data["day"] = data["hour"].map(lambda x: str(x)[4:6])
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    #此处可以考虑将组合特征的源特征删掉，对比效果
    data = data.drop(["id", "hour", "C15", "C16"], axis=1)
    return data

datapath = "."
trainfile = os.path.join(datapath ,"train_sample.csv")

df_train = pd.read_csv(trainfile,dtype={"C15":str,"C16":str})
#df_test= pd.read_csv(testfile,dtype={"C15":str,"C16":str})
df_train = create_feature(df_train)
#df_test = create_feature(df_test)
#删掉组合特征的原始特征，为了省内存
df_train = df_train.drop(["app_id","site_id","device_model"],axis=1)
# specify parameters via map
param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 4, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}
num_round = 10
y_all = df_train["click"]
df_train = df_train.drop(["click"],axis=1)

le = preprocessing.LabelEncoder()
columns_id = ["site_domain","site_category","app_domain","app_category","device_id","device_ip","app_site_id","app_site_id_model","size"]

for columnname in columns_id:
    df_train[columnname]= le.fit_transform(df_train[columnname])

columns = df_train.columns
print(columns)
X_all = df_train.values
#print(X_all.shape)
#print(y_all)
#划分数据集
print("划分数据集")
#x_train, x_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.2, random_state = 2018)

dtrain = xgb.DMatrix(X_all,y_all,feature_names=columns)
#dtest = xgb.DMatrix(x_val,y_val)
bst = xgb.train(param, dtrain, num_round)
#pred_leaf=True,
'''

test_preds = bst.predict(dtest,pred_leaf=False)
print(test_preds.shape)

test_predictions = [round(value) for value in test_preds]
y_test = dtest.get_label()
train_accuracy = accuracy_score(y_test, test_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
'''
# 显示重要特征
plot_importance(bst)
plt.savefig("feature_importtance.png")
#保存xgboost模型数据
print("saving xgboost model")
joblib.dump(bst,"xgb_ctr_joblib.dat")

#plt.show()
#得到新特征
x_train_feature = bst.predict(dtrain,pred_leaf=True)
#x_test_feature = bst.predict(dtest,pred_leaf=True)
#新特征进行标准化
x_train_feature_scale = scale(x_train_feature)
print(x_train_feature_scale)
#new_train_feature = DataFrame(x_train_feature_scale)
#new_test_feature = DataFrame(x_test_feature)
print(x_train_feature_scale.shape)


#对新特征使用onehot编码
enc = OneHotEncoder()
#new_train_feature_onehot = enc.fit_transform(x_train_feature_scale).toarray()
#new_test_feature_onehot = enc.fit_transform(new_test_feature).toarray()
del x_train_feature
gc.collect()
#GBDT生成的特征划分数据集
x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(x_train_feature_scale, y_all, test_size=0.2, random_state=2018)
#进行LR预测
# 定义LR模型
lr = LogisticRegression()
lr.fit(x_train_lr,y_train_lr)

y_pred_xgblr1_proba = lr.predict_proba(x_train_lr)
y_pred_xgblr = lr.predict(x_train_lr)
tr_logloss = log_loss(y_train_lr,y_pred_xgblr1_proba[:,1])
print("train logloss:",tr_logloss)
val_logloss = log_loss(y_val_lr,lr.predict_proba(x_val_lr)[:,1])
print("validate logloss:",val_logloss)
#print(y_pred_xgblr)
#保存logRegression
print("saving LogRegression Model data")
joblib.dump(lr,"ctr_lr_joblib.dat")
#开始预测
print("用测试集数据进行预测。")
testfile = os.path.join(datapath,"test_sample")
df_test = pd.read_csv(testfile)

#写入结果文件中
print("Write result to test file.")
'''
df_test_newfeature = create_feature(df_test)
X_test = df_test.values

bst.predict(X_test,pred_leaf=True)
lr.fit()
res = pd.read_csv("test_sample")
y_pred  = [0]
res["click"] = y_pred
outputfilename = os.path.join(datapath,"test_predict.csv")
with open(outputfilename) as outputfile:
    res["id"] = res["id"].apply(lambda x: '{:.0f}'.format(x))
    res.to_csv(outputfile,index=False,header=True)
'''