# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.path.append("Functions/Licheng")
sys.path.append("Functions/Licheng/dijkstar")
from jumpDays import *

#def obtainScore(pathList, windGraph):
#    timeLen = pathList.shape[0]
#    threshold = 15
#    seg = len(pathList)/30
#    flag = False
#    for j in range(seg):   
#        for i in range(30):
#            if(windGraph[j,pathList[i, 0], pathList[i, 1]] >= threshold):
#                print("die " + str(windGraph[j,pathList[i, 0], pathList[i, 1]]) +" " + str(j)+ " " +str(i))
#                flag = True
#    #            plt.scatter(pathList[i, 1], pathList[i, 0], marker='o', s=100, c = 'gold', zorder=10) 
#    return 1440 if flag else (timeLen - 1) * 2

submitfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\submitResult.csv"
weatherfile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"
def obtainScore(submitfile, weatherfile,cityLocFile,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,threshold = 15):
    chunksize = xsize * ysize
    pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])
    windGraph = np.zeros((18,xsize,ysize))
    Score = []
    for dayNum in [3]: #range(1,maxDay+1):
        df = pd.read_csv(weatherfile, chunksize = chunksize)
        df = jumpDays(df, dayNum-1, chunksize)
        for _ in range(18):
            windGra = df.get_chunk(chunksize)["wind"]
            windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
        for city in [5]: #range(2,maxCity+1):
            pathpiece = pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["x","y"]].reset_index(drop = True)
            return pathpiece
            seg = pathpiece.shape[0]/30
            flag = False
            for j in range(seg):   
                for i in range(30):
                    if(windGraph[j,pathpiece["x"][i], pathpiece["y"][i]] >= threshold):
                        print("die " + str(windGraph[j,pathpiece["x"][i], pathpiece["y"][i]]) +" " + str(j)+ " " +str(i))
                        flag = True          
            Score += [1440 if flag else (pathpiece.shape[0] - 1) * 2]            
    return Score   
pathpiece = obtainScore(submitfile, weatherfile,cityLocFile)