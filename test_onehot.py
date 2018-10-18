#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

import datetime
import seaborn as sns
import matplotlib.pyplot as plt
datapath = "."
trainfile = os.path.join(datapath ,"train_sample.csv")
chunks = pd.read_csv(trainfile,chunksize=10000)
for df in chunks:
    print(df.head())
    print(df.shape)
    enc = OneHotEncoder()
    x_train = df.drop(["id","click"],axis=1)
    enc.fit(x_train)
    x_train_tansform = enc.transform(x_train).toarray()
    print(x_train_tansform.shape)
    print(x_train_tansform)


