#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
datapath = "."
#print(datapath)
trainfile = os.path.join(datapath ,"train.csv")
testfile = os.path.join(datapath,"test")
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
        index +=1
        print(index)
        #print(chunk.columns)

        count = count + chunk.shape[0]
    print(count)
# #测试数据生成微小数据集
# df_test = pd.read_csv(testfile,chunksize=100000)
# print("生成测试数据微小数据集")
# index = 0
# with open("test_sample","a") as testsamplefile:
#     for chunk in df_test:
#         sample_test = chunk.sample(n=None,frac=0.1,axis=0)
#         sample_test["id"] = sample_test["id"].apply(lambda x: '{:.0f}'.format(x))
#         if index ==0:
#             sample_test.to_csv(testsamplefile,index=False,header=True)
#         else:
#             sample_test.to_csv(testsamplefile, index=False, header=False)
#         index +=1
#     print("测试集完成。")