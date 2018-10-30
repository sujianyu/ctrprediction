import datetime
#import xlearn as xl
import lightgbm as lgb
from lightgbm.sklearn import  LGBMRegressor
import pandas as pd
from category_encoders.hashing import HashingEncoder

df_X = pd.DataFrame([1,2,3,4,1,2,4,5,8,7,66,2,24,5,4,1,2,111,1,31,3,23,13,24],columns=list("A"))

he = HashingEncoder(cols=["A"],return_df=True)
df_X = he.fit_transform(df_X)
print(df_X.head())

