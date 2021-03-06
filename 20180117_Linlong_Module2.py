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
#from Path_design import *
#from Path_design_Update import *
from obtainScore import *
from Data_convert import *
from Path_generator import *

#trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
#trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
#testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"
#cityLocFile = "Data\CityData.csv"
#testTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
#submitPath = "Data\submitResult.csv"


trainPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\lwuag\Desktop\TianchiData\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTesting_201712.csv"
cityLocFile = "Data\CityData.csv"
testTrueFile = "C:\Users\lwuag\Desktop\TianchiData\predict_model_2.csv"
submitPath = "Data\submitResult_Linlong_20180129_segment0.csv"


cityLoc = pd.read_csv(cityLocFile)
xCity = cityLoc['xid']
yCity = cityLoc['yid']
file = testTrueFile
xsize = 548
ysize = 421
maxDay = 5
maxCity = 10
hourNum = 18
chunksize = xsize * ysize

block = []
windGraph = np.zeros((hourNum, xsize, ysize))
#fullScore = []
for dayNum in [3]: #range(1, maxDay + 1):
    print(dayNum)
    df = pd.read_csv(file, chunksize = chunksize)
    df = jumpDays(df, dayNum-1, chunksize)
    
    windGra = df.get_chunk(chunksize)["wind"]
    for _ in range(18):
        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
#    for _ in range(18):
#        windGra = df.get_chunk(chunksize)["wind"]
#        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
#    print(windGraph[0, xCity[0], yCity[0]])
    windGraph[0, xCity[0], yCity[0]] = min(windGraph[0, xCity[0], yCity[0]], 14.9)        
    star_point = xCity[0] * ysize + yCity[0]
    for cityNum in range(1, maxCity + 1):
        print(cityNum)
        thre_wind = 15
        height = 0
        Pathinfo = Path_generator(windGraph, xCity[0], yCity[0], xCity[cityNum], yCity[cityNum], thre_wind, height)     
#            Score = obtainScore(Pathinfo, windGraph)
        (string, des_n_day) = submitFormat(dayNum+5, cityNum, Pathinfo)
        block += list(np.concatenate((des_n_day, string, Pathinfo), axis = 1))
#            Score = 1440
#        print Score
#        fullScore += [Score]

block = np.asarray(block)
#%%
df_b = pd.DataFrame(block)
df_b.to_csv(submitPath, header=None,index = False)