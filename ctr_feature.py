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

group = df["click"].groupby(df["device_type"])
print(group.mean())
print(df["C1"].value_counts())
print(df["C15"].value_counts())
print(df["C16"].value_counts())

x_train = df.drop(["id","click"],axis = 1)
