# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
sys.path.append("Functions/Licheng")
from jumpDays import *
import matplotlib.pyplot as plt

def getWindGraph(weatherfile,dayNum,hourNum = 18,xsize = 548,ysize = 421):
    windGraph = np.zeros((hourNum,xsize,ysize))
    chunksize = xsize * ysize
    df = pd.read_csv(weatherfile, chunksize = chunksize)
    df = jumpDays(df, dayNum-1, chunksize)
    for _ in range(hourNum):
        windGra = df.get_chunk(chunksize)["wind"]
        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
    return windGraph

def partition(Len,segsize=30):
    # seg: int, segsize: int,
    # return: 1d list
    seg = []
    while(Len>=30):
        seg += [30]
        Len -= 30
    if(Len > 0):
        seg += [Len]
    return seg

def getpathpiece(pathfile,city,dayNum):
    return pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["x","y"]].reset_index(drop = True)

def explode_or_not(windGraph,pathpiece,threshold,xCity,yCity,city,seg):
    flag = False
    for j in range(len(seg)): 
        for i in range(seg[j]):
            # whenever wind over 15, explode
            if(windGraph[j,int(pathpiece["x"][j*30+i]), int(pathpiece["y"][j*30+i])] >= threshold):
                print("dead on hour: " +str(j+3)+ ", minute: " +str(i*2) + ", wind: "+ \
                    str(windGraph[j,int(pathpiece["x"][j*30+i]), int(pathpiece["y"][j*30+i])]))
                flag = True
    # not arriving destination, explode  
    j = len(seg)-1
    i = seg[j] - 1
    if pathpiece["x"][j*30+i]!= xCity[city] or pathpiece["y"][j*30+i] != yCity[city]:
        flag = True
    return flag

def obtainScore(submitfile, weatherfile,cityLocFile,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,threshold = 15):
    cityloc = pd.read_csv(cityLocFile)
    xCity = cityloc['xid']
    yCity = cityloc['yid']
    pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])
    Score = []
    print "starting...."
    print "\n"
    for dayNum in range(1,maxDay+1):
        windGraph = getWindGraph(weatherfile,dayNum)
        for city in range(1,maxCity+1):
            pathpiece = getpathpiece(pathfile,city,dayNum)
            Len = pathpiece.shape[0]
            if Len == 0:
                Score += []
            else:
                flag = explode_or_not(windGraph,pathpiece,threshold,xCity,yCity,city,partition(Len))    
                score = 1440 if flag else (pathpiece.shape[0] - 1) * 2                                     
                print "dayNum: "+str(dayNum)+", city: "+str(city)+", score: "+str(score)
                print "==========================="
                Score += [score]
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
    for dayNum in range(1,maxDay+1):
        df = pd.read_csv(weatherfile, chunksize = chunksize)
        df = jumpDays(df, dayNum-1, chunksize)
        for _ in range(18):
            windGra = df.get_chunk(chunksize)["wind"]
            windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
        for city in range(1,maxCity+1):
            pathpiece = pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["x","y"]].reset_index(drop = True)
            Len = pathpiece.shape[0]
            if(Len == 0):  continue
            seg = []
            while(Len>=30):
                seg += [30]
                Len -= 30
            if(Len > 0):
                seg += [Len]
            for j in range(min(len(seg),18)):  
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
    

