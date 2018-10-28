#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
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
plt.switch_backend('agg')
from sklearn.preprocessing import scale
import joblib
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score
import gc
# 计算分类正确率
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from MeanEncoder import MeanEncoder
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

def time_period(hour):
    period = 0
    hour = int(hour)
    if hour>0 and hour <6:
        period = 1
    elif hour>=6 and hour<11 :
        period = 2
    elif hour>=11 and hour<13:
        period = 3
    elif hour>13 and hour<17:
        period = 4
    elif hour>=17 and hour<24:
        period = 5
    return period

def create_feature(data):
    data["size"] = data["C15"] * data["C16"]
    # 将hour列拆分为
    data["hour1"] = data["hour"].map(lambda x: str(x)[6:8])
    data["day"] = data["hour"].map(lambda x: str(x)[4:6])
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
    data["time_period"] = data["hour1"].map(lambda x:time_period(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    return data

sample_datapath = "/data/sujianyu/ctrsample/"
#train_datapath = "/data/barnett007/ctr-data/"
sample_filename = "train_sample2.csv"
#train_filename = "train.csv"
output = "/output"
#本机运行时的路径
output = "output"
sample_datapath = "./data"
num_round = 5
n_components = 0.75
#trainfile = os.path.join(train_datapath ,train_filename)
trainfile = os.path.join(sample_datapath,sample_filename)
df_train = pd.read_csv(trainfile)
df_train = create_feature(df_train)
columns = df_train.columns
print(columns)
# specify parameters via map
param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 4, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}

y_all = df_train["click"]
x_train = df_train.drop(["id","click","hour","C15","C16"],axis=1)
#le = preprocessing.LabelEncoder()

columns_me_id = ["site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","C14","C17","C19","C20","C21","hour1","app_site_id","app_site_id_model"]
columns_onehot_id = ["C1","banner_pos","device_type","device_conn_type","C18","size","time_period","day","weekday"]
me = MeanEncoder(columns_me_id,target_type='classification')
enc = OneHotEncoder()
pca = PCA(n_components=n_components)
x_onehot = df_train[columns_onehot_id]
x_me = df_train[columns_me_id]
x_train_onehot = enc.fit_transform(x_onehot)
x_train_me = me.fit_transform(x_me,y_all).values
print(x_train_onehot.shape)
print(x_train_me.shape)
X_all = sparse.hstack((x_train_onehot,x_train_me))
print(X_all)
sys.exit(0)
#print(y_all)
#划分数据集
print("split dataset.")
x_train, x_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.2, random_state = 2018)

dtrain = xgb.DMatrix(x_train, y_train, feature_names=columns)
dtest = xgb.DMatrix(x_val,y_val,feature_names=columns)
watchlist1 = [(dtrain,'train'),(dtest,'test')]
bst = xgb.train(param, dtrain, num_round,early_stopping_rounds=10,evals=watchlist1)
#pred_leaf=True,

del X_all
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
plt.savefig(os.path.join(output,"feature_importtance.png"))
#保存xgboost模型数据
print("saving xgboost model")
joblib.dump(bst,os.path.join(output,"xgb_ctr_joblib.dat"))

#plt.show()
#得到新特征

x_train_feature = bst.predict(dtrain,pred_leaf=True)
x_val_feature = bst.predict(dtest,pred_leaf=True)
del dtrain
del dtest
del x_train
del x_val
gc.collect()
#x_test_feature = bst.predict(dtest,pred_leaf=True)
#新特征进行标准化
#x_train_feature_scale = scale(x_train_feature)
#print(x_train_feature_scale)
#new_train_feature = DataFrame(x_train_feature_scale)
#new_test_feature = DataFrame(x_test_feature)
print("Train data shpape")
print(x_train_feature.shape)
print("Validate data shape")
print(x_val_feature.shape)

#对新特征使用onehot编码,如果用MeanEncoder试试？
enc = OneHotEncoder()
x_train_feature_onehot = enc.fit_transform(x_train_feature).toarray()
x_val_feature_onehot = enc.transform(x_val_feature).toarray()

#此处试着用MeanEncoder进行编码
df_train_feature = pd.data
'''
此处应该用PCA降维操作，限于内存不够大，只进行一般的标准化。
'''
#此处onehot，特征急剧增加，必须进行降维操作
#设置PCA参数
print("Run PCA")


x_train_feature_pca = pca.fit_transform(x_train_feature_onehot)
print(x_train_feature_pca.shape)
print("Validate PCA")
x_val_feature_pca = pca.transform(x_val_feature_onehot)


#new_test_feature_onehot = enc.fit_transform(new_test_feature).toarray()
del x_train_feature
gc.collect()
print(x_train_feature_onehot.shape)
#GBDT生成的特征划分数据集
#x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(x_train_feature_onehot, y_train, test_size=0.2, random_state=2018)
#进行LR预测
# 定义LR模型
lr = LogisticRegression()
lr.fit(x_train_feature_pca,y_train)


y_train_proba = lr.predict_proba(x_train_feature_pca)
y_val_proba = lr.predict_proba(x_val_feature_pca)
y_train_pred = lr.predict(x_train_feature_pca)

#利用验证集验证模型效果
y_val_pred = lr.predict(x_val_feature_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val,y_val_pred)
print("Train accuracy_score: ",train_accuracy)
print("Validate accuracy_score:",val_accuracy)

val_auc = roc_auc_score(y_val,y_val_pred)
print("Validate auc:",val_auc)

tr_logloss = log_loss(y_train,y_train_proba)
print("train logloss:",tr_logloss)
val_logloss = log_loss(y_val,y_val_proba)
print("validate logloss:",val_logloss)
#print(y_pred_xgblr)
#保存logRegression
print("saving LogRegression Model data")
joblib.dump(lr,os.path.join(output,"ctr_lr_joblib.dat"))
'''
#开始预测
print("用测试集数据进行预测。")
testfile = os.path.join(datapath,"test_sample")
df_test = pd.read_csv(testfile)

#写入结果文件中
#print("Write result to test file.")

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