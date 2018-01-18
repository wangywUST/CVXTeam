# -*- coding: utf-8 -*-

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

import pandas as pd
import sys
import numpy as np
sys.path.append("Functions/Licheng")
sys.path.append("Functions/Licheng/dijkstar")
from jumpDays import *
import matplotlib.pyplot as plt

def obtainScore(submitfile, weatherfile,cityLocFile,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,threshold = 15):
    cityloc = pd.read_csv(cityLocFile)
    xCity = cityloc['xid']
    yCity = cityloc['yid']
    chunksize = xsize * ysize
    pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])
    windGraph = np.zeros((18,xsize,ysize))
    Score = []
    for dayNum in [5]:#range(1,maxDay+1):
        df = pd.read_csv(weatherfile, chunksize = chunksize)
        df = jumpDays(df, dayNum-1, chunksize)
        for _ in range(18):
            windGra = df.get_chunk(chunksize)["wind"]
            windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
        for city in [9]:#range(1,maxCity+1):
            pathpiece = pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["x","y"]].reset_index(drop = True)
            Len = pathpiece.shape[0]
            if(Len == 0):
                Score += [1440]
                continue
            seg = []
            while(Len>=30):
                seg += [30]
                Len -= 30
            if(Len > 0):
                seg += [Len]
            flag = False
            for j in range(len(seg)): 
                for i in range(seg[j]):
                    if(windGraph[j,int(pathpiece["x"][j*30+i]), int(pathpiece["y"][j*30+i])] >= threshold):
                        print("die " + str(windGraph[j,int(pathpiece["x"][j*30+i]), int(pathpiece["y"][j*30+i])]) +" " + str(j)+ " " +str(i))
                        flag = True
                if(j == len(seg)-1):
                    if pathpiece["x"][j*30+i]!= xCity[city] or pathpiece["y"][j*30+i] != yCity[city]:
                        flag = True                  
            Score += [1440 if flag else (pathpiece.shape[0] - 1) * 2]
    return Score  

def plotweather(submitfile, weatherfile,cityLocFile,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,threshold = 15):
    cityloc = pd.read_csv(cityLocFile)
    xCity = cityloc['xid']
    yCity = cityloc['yid']
    chunksize = xsize * ysize
    pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])
    windGraph = np.zeros((18,xsize,ysize))
    x = np.linspace(1, xsize, xsize)
    y = np.linspace(1, ysize, ysize)
    X,Y = np.meshgrid(y, x)
    for dayNum in [3]:#range(1,maxDay+1):
        df = pd.read_csv(weatherfile, chunksize = chunksize)
        df = jumpDays(df, dayNum-1, chunksize)
        for _ in range(18):
            windGra = df.get_chunk(chunksize)["wind"]
            windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
        for city in [9]:#range(1,maxCity+1):
            pathpiece = pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["x","y"]].reset_index(drop = True)
            Len = pathpiece.shape[0]
            if(Len == 0):  continue
            seg = []
            while(Len>=30):
                seg += [30]
                Len -= 30
            if(Len > 0):
                seg += [Len]
            for j in range(len(seg)):  
                windGraph1 = windGraph[j, :, :].copy()
                windGraph1[windGraph[j, :, :] >= 15] = 10
                windGraph1[windGraph[j, :, :] < 15] = 0
#                CS = plt.contour(X, Y, windGraph1, levels = [10],colors=('k',),linestyles=('-',),linewidths=(1,))

                plt.scatter(yCity[1:11], xCity[1:11], marker='x', s=50, c = 'gold', zorder=10)
                plt.scatter(yCity[0], xCity[0], marker='*', s=50, c = 'gold', zorder=10)
                plt.scatter(pathpiece["y"], pathpiece["x"], marker='x', s=1, c = 'gold', zorder=10)
                plt.scatter(pathpiece["y"][sum(seg[:j]):sum(seg[:j+1])], pathpiece["x"][sum(seg[:j]):sum(seg[:j+1])], marker='x', s=1, c = 'g', zorder=10)
                
                CSF = plt.contourf(X, Y, windGraph1, 8, alpha=.95, cmap=plt.cm.Greys)
#                plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels
                plt.colorbar(CSF, shrink=0.8, extend='both')
                
                plt.title(str(dayNum+5) + ' Day ' + str(city) + ' City ' + str(j + 3) + ' Hour ')
                plt.savefig('Figure/Licheng/' + str(dayNum+5) + ' Day ' + str(city) + ' City ' + str(j + 3) + ' Hour.png')
                plt.clf()
        
def cityfilter(inputfile,outputfile):
    record = pd.DataFrame(columns = ["city","Day","Time","x","y"])
    pathfile = pd.read_csv(inputfile,header = None, names = ["city","Day","Time","x","y"])
    pathpiece1 = pathfile.loc[pathfile["city"] == 2].reset_index(drop = True)
    pathpiece2 = pathfile.loc[pathfile["city"] == 9].reset_index(drop = True)
    record = record.append(pathpiece1, ignore_index = True)
    record = record.append(pathpiece2, ignore_index = True)
    record.to_csv(outputfile,header=None,index = False)
    

