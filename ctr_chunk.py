#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os

def labelencoder(df,columname):
    column_dict = {}
    key_value = 0
    column_list = df[columname].value_counts().keys().tolist()
    for column in column_list:
        if column_dict.get(column) == None:
            key_value += 1
            column_dict[column] = key_value
    return column_dict

datapath = "."
chunksize = 100000
trainfile = os.path.join(datapath ,"train_sample2.csv")
reader = pd.read_csv(trainfile,iterator=True)
loop =True
site_id_dict = {}
chunknum = 0
key_value = 0
with open("train_newfeature.csv","a") as newfile:
    while loop:
        try:
            print("loop %d" % chunknum)
            df = reader.get_chunk(chunksize)
            site_id_dict = labelencoder(df,"site_id")
            df.replace(site_id_dict)
            site_domain_dict = labelencoder(df,"site_domain")
            df.replace(site_domain_dict)
            site_category_dict = labelencoder(df,"site_category")
            df.replace()
            df["id"] = df["id"].apply(lambda x: '{:.0f}'.format(x))
            if chunknum == 0:
                df.to_csv(newfile, index=False, header=True)
            else:
                df.to_csv(newfile, index=False, header=False)
            chunknum = chunknum + 1
        except StopIteration:
                loop = False
                print("Iteration is stopped.")
    print(site_id_dict)