#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
import sys,os
import xgboost as xgb
import lightgbm as lgb
import datetime
from  sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import plot_importance
from matplotlib import pyplot as plt
import joblib
import gc
# 计算分类正确率
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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
    data["hour1"] = np.round(data.hour % 100)
    data["day"] = np.round(data.hour % 10000 / 100)
    data['day_hour'] = (data.day.values - 21) * 24 + data.hour1.values
    data['day_hour_prev'] = data['day_hour'] - 1
    data['day_hour_next'] = data['day_hour'] + 1
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
    data["time_period"] = data["hour1"].map(lambda x:time_period(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    return data

def leaftomatrx(y_pred,num_leaf):
    transformed_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
        transformed_matrix[i][temp] += 1
    return transformed_matrix
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


df_reader = pd.read_csv(trainfile,chunksize=100000)
gbm = None
num_leaf = 36
params = {
        'task': 'train',
        'application': 'regression',
        'boosting_type': 'gbdt',
        'learning_rate': 0.2,
        'num_leaves': num_leaf,
        'num_threads': 4,
        'tree_learner': 'feature',
        'min_data_in_leaf': 100,
        'metric': ['l1','l2','rmse'],  # l1:mae, l2:mse
        'max_bin': 255,
        'num_trees': 300
    }
i=1
for df_train in df_reader:
    df_train = create_feature(df_train)
    y_all = df_train["click"]
    x_train = df_train.drop(["id", "click", "hour", "C15", "C16"], axis=1)

    # le = preprocessing.LabelEncoder()
    columns_objectid = ["site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category", "device_id",
                        "device_ip", "device_model", "app_site_id", "app_site_id_model"]

    # columns_me_id = ["site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","C14","C17","C19","C20","C21","hour1","app_site_id","app_site_id_model"]
    # columns_onehot_id = ["C1","banner_pos","device_type","device_conn_type","C18","size","time_period","day","weekday"]
    me = MeanEncoder(columns_objectid, target_type='classification')
    # enc = OneHotEncoder()
    # x_onehot = df_train[columns_onehot_id]
    # x_me = df_train[columns_me_id]
    # x_train_onehot = enc.fit_transform(x_onehot)
    # 删除object类型特征
    x_train_me = me.fit_transform(x_train, y_all).drop(columns_objectid, axis=1)
    columns = x_train_me.columns
    print(x_train_me.info())
    # print(x_train_me.head())
    print(x_train_me.shape)
    # X_all = sparse.hstack((x_train_onehot,x_train_me))
    # print(y_all)
    # 划分数据集
    print("split dataset.")


    x_train, x_val, y_train, y_val = train_test_split(x_train_me, y_all, test_size = 0.2, random_state = 2018)
    #训练数据集
    lgb_train = lgb.Dataset(x_train,y_train)
    #测试数据集
    lgb_val = lgb.Dataset(x_val,y_val)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_val,
                    init_model=gbm,  # 如果不为空，就是继续上次的训练。
                    #feature_name=columns,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True)  # 增量训练
    score_train = dict([(s[1],s[2]) for s in gbm.eval_train()])
    score_valid = dict([(s[1],s[2]) for s in gbm.eval_valid()])
    print('train:mae=%.4f, mse=%.4f, rmse=%.4f' % (score_train['l1'], score_train['l2'], score_train['rmse']))
    print('valid:mae=%.4f, mse=%.4f, rmse=%.4f' % (score_valid['l1'], score_valid['l2'], score_valid['rmse']))


    y_train_pred_leaf = gbm.predict(x_train,pred_leaf=True)
    y_train_pred = gbm.predict(x_train)

    print("y_train_pred:",y_train_pred)
    #取训练集预测叶子数据
    transformed_train_matrix = leaftomatrx(y_train_pred_leaf,num_leaf)
    print(transformed_train_matrix)
    print('Calculate val feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))
    print('Feature importances:', list(gbm.feature_importance("gain")))

    #取验证集叶子数据
    y_val_pred = gbm.predict(x_val, pred_leaf=True)

    # feature transformation and write result
    print('Writing transformed val data')
    transformed_val_matrix = leaftomatrx(y_val_pred,num_leaf)
    #对结果进行
    del transformed_train_matrix
    del transformed_val_matrix
    gc.collect()

    sys.exit(0)
print("Starting SGD")




