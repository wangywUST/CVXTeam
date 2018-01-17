import sys
sys.path.append("Functions")
from heuristicSquareAlg import *
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
file = trainTrueFile

xsize = 548
ysize = 421

chunksize = xsize * ysize
df = pd.read_csv(file, chunksize = chunksize)
#print(df.get_chunk(chunksize))
iniLoc = np.asarray([xCity[0], yCity[0]])
newLoc = iniLoc.copy()
print("Now, the location is ",newLoc)
print("Wind Speed is",windGra[newLoc[0], newLoc[1]])
for i in range(18):
    lastLoc = newLoc.copy()
    df_block = df.get_chunk(chunksize)
    windGra = df_block["wind"].values.reshape(xsize,ysize)
    (newLoc, flag) = heuristicSquareAlg(windGra, lastLoc, np.asarray([xCity[3], yCity[3]]), 60)
    if(flag == True):
        break
    print("Now, the location is ",newLoc)
    print("Wind Speed is",windGra[newLoc[0], newLoc[1]])
    
print("Now, the location is ",newLoc)
print("Wind Speed is",windGra[newLoc[0], newLoc[1]])