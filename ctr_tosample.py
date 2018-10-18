#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
datapath = "."
#print(datapath)
trainfile = os.path.join(datapath ,"train.csv")

df = pd.read_csv(trainfile,chunksize=100000)
#数据随机采样
'''
原始数据集较大，现有内存无法支持。因此先期进行采样
'''
count  = 0
index = 0
with open("train_sample2.csv","a") as samplefile:
    for chunk in df:
        sample = chunk.sample(n=None,frac=0.010,axis=0)
        sample["id"] = sample["id"].apply(lambda x: '{:.0f}'.format(x))
        if index==0:
            sample.to_csv(samplefile, index=False, header=True)
        else:
            sample.to_csv(samplefile, index=False, header=False)
        index = index + 1
        print(index)
        #print(chunk.columns)

        count = count + chunk.shape[0]

print(count)