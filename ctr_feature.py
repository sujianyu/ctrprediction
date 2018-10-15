#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
datapath = "."
trainfile = os.path.join(datapath ,"train_sample.csv")

df = pd.read_csv(trainfile)
result = df[["C14","C15","C16","C17","C18","C19","C20","C21"]].describe()

with open("train_sample_describe.csv","w") as f:
    pd.SparseDataFrame(result).to_csv(f)

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
x_train = df.drop(["id","click"],axis = 1)
