#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os

def labelencoder(column_list,column_dict):

    key_value = 0
    #column_list = df[columname].value_counts().keys().tolist()
    for column in column_list:
        if column_dict.get(column) == None:
            key_value += 1
            column_dict[column] = key_value
    return column_dict

datapath = "."
chunksize = 100000
trainfile = os.path.join(datapath ,"train.csv")
reader = pd.read_csv(trainfile,iterator=True)
loop =True
columns_dict= {}
column_dict = {}
site_id_dict = {}
site_domain_dict = {}
site_category_dict = {}
app_id_dict = {}
app_domain_dict= {}
app_category_dict = {}
device_id_dict = {}
device_ip_dict = {}
device_model_dict = {}

chunknum = 0
key_value = 0
key_column= ["site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_model"]
with open("train_newfeature.csv","a") as newfile:
    while loop:
        try:
            print("loop %d" % chunknum)
            df = reader.get_chunk(chunksize)
            for column_name in key_column:
                print(column_name)
                column_dict = columns_dict.get(column_name)
                if column_dict == None :
                    column_dict = {}
                    columns_dict[column_name] = column_dict
                labelencoder(df[column_name].value_counts().keys().tolist(),column_dict)
                #column_dict.update(column_dict_update)
                df[column_name].replace(column_dict,inplace=True)
                #columns_dict[column_name] = column_dict
            print("saving chunk")
            df["id"] = df["id"].apply(lambda x: '{:.0f}'.format(x))
            if chunknum == 0:
                df.to_csv(newfile, index=False, header=True)
            else:
                df.to_csv(newfile, index=False, header=False)
            chunknum = chunknum + 1
        except StopIteration:
                loop = False
                print("Iteration is stopped.")
