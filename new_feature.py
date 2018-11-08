#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用MearEncoder 编码法，生成新特征并保存为特征文件。
"""
from MeanEncoder import MeanEncoder
import numpy as np
import pandas as pd
import sys,os
import datetime
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
def time_period(hour):
    period = 0
    hour = int(hour)
    if hour>0 and hour <6:
        period = 1
    elif hour>=6 and hour<11 :
        period = 2
    elif hour>=11 and hour<13:
        period = 3
    elif hour>13 and hour<17:
        period = 4
    elif hour>=17 and hour<24:
        period = 5
    return period
def create_feature(data):
    data["size"] = data["C15"] * data["C16"]
    # 将hour列拆分为
    data["hour1"] = data["hour"].map(lambda x: str(x)[6:8])
    data["day"] = data["hour"].map(lambda x: str(x)[4:6])
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
    data["time_period"] = data["hour1"].map(lambda x:time_period(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    #此处可以考虑将组合特征的源特征删掉，对比效果
    #data = data.drop(["hour"], axis=1)
    return data
if __name__ == "__main__":
    train_file_name = "train.csv"
    test_file_name = "test"
    new_feature_filename = "test_new_featrue.csv"
    data_path = "./data"
    output = "output"
    df_reader = pd.read_csv(os.path.join(data_path,test_file_name),chunksize=200000)
    count = 0
    index = 0
    with open(os.path.join(output, new_feature_filename), "w") as newfeature:
        for df in df_reader:
            df_newfeature = create_feature(df)
            df_newfeature["id"] = df_newfeature["id"].apply(lambda x: '{:.0f}'.format(x))
            if index == 0:
                df_newfeature.to_csv(newfeature,index=False,header=True)
            else:
                df_newfeature.to_csv(newfeature,index=False,header=False)
            index +=1
            print(index)




