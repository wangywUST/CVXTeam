# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:25:56 2018

@author: lzhaoai
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:21:22 2018

@author: ywanggp
"""

import sys
sys.path.append("Functions/Linlong")
sys.path.append("Functions/Linlong/dijkstar")
from jumpDays import *
from givePath import *
from submitFormat import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from Path_design import *
from Path_design_Update import *
from obtainScore import *
from Data_convert import *

# trainPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTraining_201712.csv"
# trainTrueFile = "C:\Users\lwuag\Desktop\TianchiData\In_situMeasurementforTraining_201712.csv"
# testPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTesting_201712.csv"
# cityLocFile = "C:\Users\lwuag\Desktop\TianchiData\CityData.csv"
# testTrueFile = "C:\Users\lwuag\Desktop\TianchiData\predict_model_2.csv"
# submitPath = "C:\Users\lwuag\Desktop\TianchiData\submitResult.csv"

trainPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\lwuag\Desktop\TianchiData\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTesting_201712.csv"
cityLocFile = "C:\Users\lwuag\Desktop\TianchiData\CityData.csv"
testTrueFile = "C:\Users\lwuag\Desktop\TianchiData\predict_model_2.csv"
submitPath = "C:\Users\lwuag\Desktop\TianchiData\submitResult.csv"

#trainPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTraining_201712.csv"
#trainTrueFile = "C:\Users\lwuag\Desktop\TianchiData\In_situMeasurementforTraining_201712.csv"
#testPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTesting_201712.csv"
#cityLocFile = "C:\Users\lwuag\Desktop\TianchiData\CityData.csv"
#testTrueFile = "C:\Users\lwuag\Desktop\TianchiData\predict_model_2.csv"
#submitPath = "C:\Users\lwuag\Desktop\TianchiData\submitResult.csv"

cityLoc = pd.read_csv(cityLocFile)
xCity = cityLoc['xid']
yCity = cityLoc['yid']
file = testTrueFile

xsize = 548
ysize = 421

maxDay = 5
maxCity = 10

chunksize = xsize * ysize
df1 = pd.read_csv(file, chunksize = chunksize)

block = []
windGraph = np.zeros((18,xsize,ysize))
fullScore = []
for dayNum in [3]: #range(1, maxDay + 1):
    df = pd.read_csv(file, chunksize = chunksize)
    df = jumpDays(df, dayNum-1, chunksize)
    for _ in range(18):
        windGra = df.get_chunk(chunksize)["wind"]
        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()

    for cityNum in [5]: #range(1, maxCity + 1):
        thre_wind = 15
        height = 0
        Pathinfo = Path_generator(windGraph, xCity[0], yCity[0], xCity[cityNum], yCity[cityNum], thre_wind, height)     
#        # original algorithm
##        Pathinfo = Path_design(windGraph, star_point, end_point, end_point, 0)
#        #updated algorithm
#        
#        try:
#            Pathinfo = Path_design_Update(Data, star_point, end_point, end_point, 0)
#        except:
#            Pathinfo = Remedy_4_no_way(Data, star_point, end_point)
#            
#        #check whether the end is the desired
#        end_pos = Pathinfo[-1]
#        end_x, end_y = index_2_xy(end_pos, Data.shape[2])
#        if end_x != xCity[cityNum] or end_y != yCity[cityNum]:
#            if len(Pathinfo) <= 540:
#                Pathinfo = []
#            else:
#                Pathinfo = Remedy_4_false_end(Data, Pathinfo, star_point, end_point)
#       
#    Pathinfo = np.asarray([[node/ysize, node%ysize] for node in Pathinfo])
    print obtainScore(Pathinfo, windGraph)