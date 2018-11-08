import datetime
#import xlearn as xl
import lightgbm as lgb
from lightgbm.sklearn import  LGBMRegressor
import pandas as pd
from FTRLp import DataGen
from FTRLp import FTRLP
# from category_encoders.hashing import HashingEncoder
#
# df_X = pd.DataFrame([1,2,3,4,1,2,4,5,8,7,66,2,24,5,4,1,2,111,1,31,3,23,13,24],columns=list("A"))
#
# he = HashingEncoder(cols=["A"],return_df=True)
# df_X = he.fit_transform(df_X)
# print(df_X.head())


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import sys,os
import pickle
datapath = "./data"
output = "output"
new_featurename = "new_featrue.csv"
train_filename = "train_sample2.csv"
test_filename = "test"
train_file= os.path.join(datapath,train_filename)
test_file = os.path.join(datapath,test_filename)
new_feature_file = os.path.join(output,new_featurename)

submission = 'ftrl1sub.csv'  # path of to be outputted submission file

num_feature = 22
columns_id = ["site_id","site_domain","app_id","app_domain","device_id","device_ip","app_site_id"]
columns_category = ["C1","site_category","banner_pos","device_type","app_category","device_model","device_conn_type","device_type","C18","C20","C14","C15","C16","C17","C19","C21","app_site_id_model"]
columns_num = ["hour","hour1","day","weekday","time_period","size"]
target = "click"

dh = DataGen(num_feature,target,columns_id,columns_category,columns_num)
dh.train(train_file)
ftrl = FTRLP(rate=10)
ftrl.partial_fit(dh,train_file)
y_predict = ftrl.predict(dh,test_file)
print(y_predict)


