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
from sklearn.preprocessing import scale
import joblib
import gc
# 计算分类正确率
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tinyenv.flags import flags
import argparse
parser = argparse.ArgumentParser()

FLAGS = flags()
print(FLAGS.iterations)
sys.exit(0)
new_featurefilename = "new_featrue.csv"
datapath = "."
output = "output"
new_featurefile = os.path.join(output ,new_featurefilename)

df_reader = pd.read_csv(trainfile,chunksize=300000,dtype={"C15":str,"C16":str})
gbm = None
params = {
        'task': 'train',
        'application': 'regression',
        'boosting_type': 'gbdt',
        'learning_rate': 0.2,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'min_data_in_leaf': 100,
        'metric': ['l1','l2','rmse'],  # l1:mae, l2:mse
        'max_bin': 255,
        'num_trees': 300
    }
i=1
for df_train in df_reader:
    df_train = create_feature(df_train)
    y_all = df_train["click"]
    df_train = df_train.drop(["click"],axis=1)
    #lightgbm 模型可以对类别型特征
    columns = df_train.columns.tolist()
    print(columns)
    X_all = df_train.values
    #print(X_all.shape)
    #print(y_all)
    #划分数据集
    print("split dataset")
    x_train, x_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.2, random_state = 2018)
    lgb_train = lgb.Dataset(x_train,y_train)
    lgb_eval = lgb.Dataset(x_val,y_val)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,  # 如果不为空，就是继续上次的训练。
                    feature_name=columns,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True)  # 增量训练
    score_train = dict([(s[1],s[2]) for s in gbm.eval_train()])
    score_valid = dict([(s[1],s[2]) for s in gbm.eval_valid()])
    print('mae=%.4f, mse=%.4f, rmse=%.4f' % (score_train['l1'], score_train['l2'], score_train['rmse']))
    print('mae=%.4f, mse=%.4f, rmse=%.4f' % (score_valid['l1'], score_valid['l2'], score_valid['rmse']))
    i+=1

