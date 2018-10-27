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

def create_feature(data):
    data["size"] = data["C15"] * data["C16"]
    # 将hour列拆分为
    data["hour1"] = data["hour"].map(lambda x: str(x)[6:8])
    data["day"] = data["hour"].map(lambda x: str(x)[4:6])
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
#    data["time_period"] = data["hour1"].map(lambda x:time_period(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    #此处可以考虑将组合特征的源特征删掉，对比效果
    data = data.drop(["hour"], axis=1)
    return data
if __name__ == "__main__":
    train_file_name = "train_sample.csv"
    data_path = "."
    output = "output"
    df = pd.read_csv(os.path.join(data_path,train_file_name))
    y_all = df["click"]
    x_all = create_feature(df.drop(["id","click"],axis=1))

    print(x_all.columns)

    category_col= ["site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","app_site_id","app_site_id_model"]
    me = MeanEncoder(category_col,target_type='classification')
    train_new = me.fit_transform(x_all,y_all)
    train_newfeature = pd.concat((train_new,y_all),axis=1)
    new_feature_filename = "new_featrue.csv"
    with open(os.path.join(output,new_feature_filename),"w") as newfeature:
        train_newfeature.drop(category_col,axis=1).to_csv(newfeature,index=False)
