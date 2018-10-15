#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
datapath = "."
trainfile = os.path.join(datapath ,"train_sample.csv")

df = pd.read_csv(trainfile)
'''
result = df[["C14","C15","C16","C17","C18","C19","C20","C21"]].describe()

with open("train_sample_describe.csv","w") as f:
    pd.SparseDataFrame(result).to_csv(f)
'''
def getweekday(date):
    '''
    :param date: YYMMDD
    :return: weekday
    '''
    year = str(date)[0:]
df.info()

print(df["click"].mean())
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
group = df["click"].groupby(df["device_type"])
print(group.mean())
print("################")
print(df["device_conn_type"].value_counts())
print("################")
print(df["C15"].value_counts())
print(df["C16"].value_counts())
print(df["C17"].value_counts())
print(df["C18"].value_counts())
print(df["C19"].value_counts())
print(df["C20"].value_counts())
print(df["C21"].value_counts())
print("################")
y_train = df["click"]
df = df.drop(["id","click"],axis=1)
df["C15"] = df["C15"].map(lambda x:str(x))
df["C16"] = df["C16"].map(lambda x:str(x))
#建立新的特征列
df["size"] = df["C15"].str.cat(df["C16"],sep="_")
#将hour列拆分为
df["hour1"] = df["hour"].map(lambda x:str(x)[6:8])
df["day"] = df["hour"].map(lambda x:str(x)[4:6])

print(df.head())