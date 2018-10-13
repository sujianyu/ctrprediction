#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys,os
datapath = "."
chunksize = 100000
trainfile = os.path.join(datapath ,"train.csv")
reader = pd.read_csv(trainfile,iterator=True)
loop =True
chunks = []
chunknum = 0
while loop:
    try:
        chunknum = chunknum + 1
        print(chunknum)
        chunk = reader.get_chunk(chunksize)
        chunks.append(chunk)
    except StopIteration:
            loop = False
            print("Iteration is stopped.")
df = pd.concat(chunks,ignore_index=True)
print(df.describe())