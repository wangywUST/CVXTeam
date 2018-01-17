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

# trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
# trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
# testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"
# cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"
# testTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
# submitPath = "C:\Users\lzhaoai\Desktop\predict_weather\submitResult.csv"


trainPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\lwuag\Desktop\TianchiData\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\lwuag\Desktop\TianchiData\ForecastDataforTesting_201712.csv"
cityLocFile = "Data\CityData.csv"
testTrueFile = "C:\Users\wangyw\Dropbox\Contest\contest\Data\predict_model_2.csv"
submitPath = "Data\submitResult.csv"


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
windGraph = np.zeros((hourNum,xsize,ysize))
fullScore = []
for dayNum in range(1, maxDay + 1):
    df = pd.read_csv(file, chunksize = chunksize)
    df = jumpDays(df, dayNum-1, chunksize)
    for _ in range(18):
        windGra = df.get_chunk(chunksize)["wind"]
        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()

    star_point = xCity[0] * ysize + yCity[0]
    for cityNum in range(1, maxCity + 1):
        end_point = xCity[cityNum] * ysize + yCity[cityNum]
        # original algorithm
#        Pathinfo = Path_design(windGraph, star_point, end_point, end_point, 0)
        #updated algorithm
        thre_wind = 15
        Data = Data_convert(windGraph, thre_wind)
        try:
            Pathinfo = Path_design_Update(Data, star_point, end_point, end_point, 0)
            Pathinfo = np.asarray([[node/ysize, node%ysize] for node in Pathinfo])
            Score = obtainScore(Pathinfo, windGraph)
            (string, des_n_day) = submitFormat(dayNum+5, cityNum, Pathinfo)
            block += list(np.concatenate((des_n_day, string, Pathinfo), axis = 1))
        except:
            Score = 1440
        print Score
        fullScore += [Score]

block = np.asarray(block)
#%%
df_b = pd.DataFrame(block)
df_b.to_csv(submitPath, header=None,index = False)