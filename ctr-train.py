#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys, os
import datetime

datapath = "/data/barnett007/ctr-data"
print(datapath)
trainfile = os.path.join(datapath, "train.csv")
print("reading train.csv")
df = pd.read_csv(trainfile)
'''
result = df[["C14","C15","C16","C17","C18","C19","C20","C21"]].describe()

with open("train_sample_describe.csv","w") as f:
    pd.SparseDataFrame(result).to_csv(f)
'''


def getweekday(x):
    '''
    :param date: YYMMDD
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
    if hour > 0 and hour < 6:
        period = 1
    elif hour >= 6 and hour < 11:
        period = 2
    elif hour >= 11 and hour < 13:
        period = 3
    elif hour > 13 and hour < 17:
        period = 4
    elif hour >= 17 and hour < 24:
        period = 5
    return period


df.info()
print("column info")
print(df["click"].mean())
# 建立新的特征列
print("create new column")
df["size"] = df["C15"] * df["C16"]
# 将hour列拆分为
df["hour1"] = df["hour"].map(lambda x: str(x)[6:8])
df["day"] = df["hour"].map(lambda x: str(x)[4:6])
df["weekday"] = df["hour"].map(lambda x: getweekday(x))
df["time_period"] = df["hour1"].map(lambda x: time_period(x))
print("################")
print(df["C1"].value_counts())
print("################")
print(df["banner_pos"].value_counts())
print("################")
print(df["site_id"].value_counts())
print("################")
print(df["site_domain"].value_counts())
print("################")
print(df["site_category"].value_counts())
print("################")
print(df["app_id"].value_counts())
print("################")
print(df["app_domain"].value_counts())
print("################")
print(df["app_category"].value_counts())
print("################")
print(df["device_ip"].value_counts())
print("################")
print(df["device_model"].value_counts())
print("################")
print(df["device_type"].value_counts())
print("################")
group = df["click"].value_counts()
print("################")
print(df["device_conn_type"].value_counts())
print("################")
print(df["C14"].value_counts())
print(df["C15"].value_counts())
print(df["C16"].value_counts())
print(df["C17"].value_counts())
print(df["C18"].value_counts())
print(df["C19"].value_counts())
print(df["C20"].value_counts())
print(df["C21"].value_counts())
print("################")
print(df["size"].value_counts())
print(df["hour1"].value_counts())
print(df["day"].value_counts())
print(df["weekday"].value_counts())
print(df["time_period"].value_counts())
