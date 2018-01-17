# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:12:16 2018

@author: ywanggp
"""

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np


trainPredFile = "C:/Users/ywanggp/Downloads/ForecastDataforTraining_201712.csv"
trainTrueFile = "C:/Users/ywanggp/Downloads/In_situMeasurementforTraining_201712.csv"
testPredFile = "C:/Users/ywanggp/Downloads/ForecastDataforTesting_201712.csv"
cityLocFile = "C:/Users/ywanggp/Downloads/CityData.csv"

cityLoc = pd.read_csv(cityLocFile)
xCity = cityLoc['xid']
yCity = cityLoc['yid']
file1 = trainPredFile
file2 = trainTrueFile

xsize = 548
ysize = 421

chunksize = xsize * ysize * 10
df1 = pd.read_csv(file1, chunksize = chunksize)
df2 = pd.read_csv(file2, chunksize = xsize * ysize)
pd.set_option('display.max_rows', None)
froNorm = []
for i in range(90):
    df1_block = df1.get_chunk(chunksize)
    windGra1 = df1_block["wind"].values.reshape(xsize * ysize, 10)
    windGra1 = np.median(windGra1, axis = 1).reshape(xsize, ysize)

    df2_block = df2.get_chunk(chunksize / 10)
    windGra2 = df2_block["wind"].values.reshape(xsize, ysize)
    print(np.linalg.norm(windGra1 - windGra2, 'fro'))
    froNorm += [np.linalg.norm(windGra1 - windGra2, 'fro')]

plt.plot(froNorm)
plt.show()