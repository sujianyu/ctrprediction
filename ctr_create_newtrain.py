#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
import datetime
datapath = "."
outputpath = "output"
#print(datapath)
trainfile = os.path.join(datapath ,"train_sample2.csv")
trainnewfile = os.path.join(outputpath,"train_new.csv")

def getweekday(x):
    '''
    :param x: YYMMDD
    :return: weekday
    '''
    sdt = str(x)
    year = int(sdt[0:2])
    month = int(sdt[2:4])
    day = int(sdt[4:6])
    dt = datetime.date(year, month, day)
    weekday = dt.weekday()
    return weekday

chunks= pd.read_csv(trainfile,chunksize=100000,dtype={"C15":str,"C16":str})
count  = 0
index = 0
with open(trainnewfile,"a") as trainnewfile:
    for df in chunks:
        df["size"] = df["C15"].str.cat(df["C16"], sep="_")
        # 将hour列拆分为
        df["hour1"] = df["hour"].map(lambda x: str(x)[6:8])
        df["day"] = df["hour"].map(lambda x: str(x)[4:6])
        df["weekday"] = df["hour"].map(lambda x: getweekday(x))
        df["site_id_domain_cat"] = df["site_id"]+ "_" + df["site_domain"] + "_" + df["site_category"]
        df["app_id_domain_cat"] = df["app_id"] + "_" + df["app_domain"] + "_" + df["app_category"]
        df["device_id_ip_mod"] = df["device_id"] + "_" + df["device_ip"] + "_" + df["device_model"]


        columns=["id", "site_id","site_domain","site_category", "app_id","app_domain","app_category", "hour", "C15", "C16"]
        df = df.drop(columns, axis=1)
        if index==0:
            df.to_csv(trainnewfile, index=False, header=True)
        else:
            df.to_csv(trainnewfile, index=False, header=False)
        index = index + 1
        print(index)

print("endding")