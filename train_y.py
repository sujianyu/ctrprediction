#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
datapath = "./data"
output = "output"
#print(datapath)
trainfile = os.path.join(datapath ,"train.csv")
testfile = os.path.join(datapath,"test")
df_reader = pd.read_csv(trainfile,chunksize=100000)
'''
将click列单拿出来用于SGDClassfier，增量训练。
'''
count  = 0
index = 0
with open(os.path.join(output,"train_y.csv"),"a") as samplefile:
    for df in df_reader:

        df_y = df["click"]
        if index==0:
            df_y.to_csv(samplefile, index=False, header=True)
        else:
            df_y.to_csv(samplefile, index=False, header=False)
        index +=1
        print(index)
        #print(chunk.columns)