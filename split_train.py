#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
原始数据集较大，现有内存无法支持。因此将大数据集拆分成小数据集。
按1000000分批次写入新文件中。
'''
import numpy as np
import pandas as pd
import sys,os
datapath = "./data"
output="output"
#print(datapath)
trainfile = os.path.join(datapath ,"train.csv")
df_reader = pd.read_csv(trainfile,chunksize=1000000)
count  = 0
index = 0
splitfilename = "train_split_"
for df in df_reader:
    index += 1
    split_train_filename = os.path.join(output,splitfilename + str(index) + ".csv")
    print("split_train_filename=",split_train_filename)
    print("saving chunk:", index)
    with open(split_train_filename,"w") as split_trainfile:

        df["id"] = df["id"].apply(lambda x: '{:.0f}'.format(x))
        df.to_csv(split_trainfile, index=False, header=True)
    print(index)