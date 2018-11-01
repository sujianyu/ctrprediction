
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
from scipy import sparse
import pandas as pd
from pandas import DataFrame
import sys,os
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

sample_datapath = "/data/sujianyu/ctrsample/"
#train_datapath = "/data/barnett007/ctr-data/"
train_filename = "train_sample2.csv"
#train_filename = "train.csv"
# 第一步：初始化模型爲None,設置模型保存路徑
model = None
datapath = "./data"
output = "output"
model_file = os.path.join(output,"nn_model.h5")
train_file = os.path.join(datapath,train_filename)
# 第二步：每次讀取10萬行數據
i = 1
columns_objectid = ["site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category", "device_id",
                        "device_ip", "device_model", "app_site_id", "app_site_id_model"]
me = MeanEncoder(columns_objectid, target_type='classification')
pd_reader = pd.read_csv(train_file,chunksize=100000)
y_logloss = []
x_index = []
for df_train in pd_reader:
    print("i=",i)
    #x_data = train[x_cols]
    #y_data = train[[y_col]]
    df_train = create_feature(df_train)
    y_all = df_train["click"]
    x_train = df_train.drop(["id", "click", "hour", "C15", "C16"], axis=1)

    x_train_me = me.fit_transform(x_train, y_all).drop(columns_objectid, axis=1)
    #print("x_train_me.shape",x_train_me.shape)
    columns = x_train_me.columns

    if model == None:
        k = len(columns)
        print("k=",k)
        model = Sequential()
        model.add(Dense(80, activation='relu', input_shape=(k,), ))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add((Dropout(0.5)))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        print('building DNN')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'mae', 'mape'])
        print("compile")
    else:
        model = load_model(model_file)
        print('Loading last one Model file.')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    history = model.fit(x=x_train_me, y=y_all, batch_size=5000, epochs=20,verbose=10000)
    print('model finished.')

    model.save(model_file)

    # 第七步：查看模型評分
    # 查看模型評分
    loss = history.history['loss'][-1]
    mse = history.history['mean_squared_error'][-1]
    mae = history.history['mean_absolute_error'][-1]
    mape = history.history['mean_absolute_percentage_error'][-1]
    print('第%2d 批次數據，loss=%.4f, mse=%.4f, mae=%.4f, mape=%.4f' % (i, loss, mse, mae, mape))

    i += 1
    x_index.append(i)
    y_logloss.append(loss)

#print(x_index)
#print(y_logloss)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_index,y_logloss)
plt.xlabel('index')
plt.ylabel('score')
plt.tight_layout()
plt.savefig(os.path.join(output,'score.png'))
plt.close()
#plt.show()
