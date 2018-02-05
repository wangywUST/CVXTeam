# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
sys.path.append("Functions/Licheng")
from jumpDays import *
import matplotlib.pyplot as plt

def get_Wind_Rain_Graph(weatherfile,dayNum,hourNum = 18,xsize = 548,ysize = 421):
    windGraph = np.zeros((hourNum,xsize,ysize))
    rainGraph = np.zeros((hourNum,xsize,ysize))
    chunksize = xsize * ysize
    df = pd.read_csv(weatherfile, chunksize = chunksize)
    df = jumpDays(df, dayNum-1, chunksize)
    for _ in range(hourNum):
        wind_rain_Gra = df.get_chunk(chunksize)[["wind","rainfall"]]
        windGraph[_,:,:] = wind_rain_Gra["wind"].values.reshape(xsize,ysize).copy()
        rainGraph[_,:,:] = wind_rain_Gra["rainfall"].values.reshape(xsize,ysize).copy()
    return windGraph,rainGraph

def partition(Len,initial):
    # seg: int, segsize: int,
    # return: 1d list
    initial_int = map(int, initial.split(":"))
    seg = [0]*(initial_int[0]-3) + [30 - initial_int[1]/2]
    Len -= (30 - initial_int[1]/2)
    while(Len>=30):
        seg += [30]
        Len -= 30
    if(Len > 0):
        seg += [Len]
    return seg

def getpathtmp(pathfile,city,dayNum):
    return pathfile.loc[(pathfile["city"] == city) & (pathfile["Day"] == dayNum + 5)][["Time","x","y"]].reset_index(drop = True)

def explode_or_not(windGraph,rainGraph,pathpiece,threshold_wind, threshold_rain,xCity,yCity,city,seg,hourNum):
    flag = False
    for j in range(min(len(seg),hourNum)): 
        for i in range(seg[j]):
            # whenever wind over 15, explode
            if(windGraph[j,int(pathpiece["x"][sum(seg[:j])+i]), int(pathpiece["y"][sum(seg[:j])+i])] >= threshold_wind or\
               rainGraph[j,int(pathpiece["x"][sum(seg[:j])+i]), int(pathpiece["y"][sum(seg[:j])+i])] >= threshold_rain):
                print("dead on hour: " +str(j+3)+ ", minute: " +str((30 - seg[j])*2 + i*2) + ", wind: "+ \
                    str(windGraph[j,int(pathpiece["x"][sum(seg[:j])+i]), int(pathpiece["y"][sum(seg[:j])+i])]) \
                    +", rain: " + str(rainGraph[j,int(pathpiece["x"][sum(seg[:j])+i]), int(pathpiece["y"][sum(seg[:j])+i])]))
                flag = True
    # not arriving destination, explode  
    if pathpiece["x"][sum(seg[:min(len(seg),hourNum)])-1]!= xCity[city] or pathpiece["y"][sum(seg[:min(len(seg),hourNum)])-1] != yCity[city]:
        print "Not ariving in destination!"
        flag = True
    return flag

def obtainScore(submitfile, weatherfile,cityLocFile,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,\
    threshold_wind = 15,threshold_rain = 4,hourNum = 18):
    cityloc = pd.read_csv(cityLocFile)
    xCity = cityloc['xid']
    yCity = cityloc['yid']
    pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])
    Score = []
    print "starting...."
    print "\n"
    for dayNum in range(1,maxDay+1):
        windGraph,rainGraph = get_Wind_Rain_Graph(weatherfile,dayNum)
        for city in range(1,maxCity+1):
            pathtmp = getpathtmp(pathfile,city,dayNum)
            Len = pathtmp.shape[0]
            if Len == 0:
                Score += []
            else:
                pathpiece,initial = pathtmp[["x","y"]],pathtmp["Time"][0]
                flag = explode_or_not(windGraph,rainGraph,pathpiece,threshold_wind, threshold_rain,xCity,yCity,city,partition(Len,initial),hourNum)    
                score = 1440 if flag else (pathpiece.shape[0] - 1) * 2                                     
                print "dayNum: "+str(dayNum)+", city: "+str(city)+", score: "+str(score)
                print "==========================="
                Score += [score]
    return Score  

def plot_func(dayNum,city,pathpiece,windGraph,rainGraph,xCity,yCity,hourNum,seg,X,Y):
    for j in range(min(len(seg),hourNum)):  
        print "Day: "+str(dayNum+5) + ", Hour: "+ str(j + 3) + ", City: " + str(city)
        windGraph_j = windGraph[j, :, :].copy()
        rainGraph_j = rainGraph[j, :, :].copy()
        feasible_j = np.zeros(windGraph_j.shape)
        feasible_j[(windGraph_j >= 15) | (rainGraph_j >= 4)] = 1
        plt.scatter(yCity[1:11], xCity[1:11], marker='x', s=50, c = 'gold', zorder=10)
        plt.scatter(yCity[0], xCity[0], marker='*', s=50, c = 'gold', zorder=10)
        plt.scatter(pathpiece["y"], pathpiece["x"], marker='x', s=1, c = 'gold', zorder=10)
        plt.scatter(pathpiece["y"][sum(seg[:j]):sum(seg[:j+1])], pathpiece["x"][sum(seg[:j]):sum(seg[:j+1])], \
            marker='x', s=1, c = 'g', zorder=10)
        CSF = plt.contourf(X, Y, feasible_j, 8, alpha=.95, cmap=plt.cm.Greys)
        plt.colorbar(CSF, shrink=0.8, extend='both')
        plt.title("Day: "+str(dayNum+5) + ", Hour: "+ str(j + 3) + ", City: " + str(city))
        plt.savefig('Figure/Licheng/' + "Day_"+str(dayNum+5) + "_City_" + str(city) + "_Hour_"+ str(j + 3) \
             + '.png')
        plt.clf()
        print "Done!"

def plot_func_ref(dayNum,windGraph,rainGraph,xCity,yCity,hourNum,X,Y):
    for j in range(hourNum):  
        print "Day: "+str(dayNum+5) + ", Hour: "+ str(j + 3)
        windGraph_j = windGraph[j, :, :].copy()
        rainGraph_j = rainGraph[j, :, :].copy()
        feasible_j = np.zeros(windGraph_j.shape)
        feasible_j[(windGraph_j >= 15) | (rainGraph_j >= 4)] = 1
        plt.scatter(yCity[1:11], xCity[1:11], marker='x', s=50, c = 'gold', zorder=10)
        plt.scatter(yCity[0], xCity[0], marker='*', s=50, c = 'gold', zorder=10)
        CSF = plt.contourf(X, Y, feasible_j, 8, alpha=.95, cmap=plt.cm.Greys)
        plt.colorbar(CSF, shrink=0.8, extend='both')
        plt.title("Day: "+str(dayNum+5) + ", Hour: "+ str(j + 3))
        plt.savefig('Figure/Licheng/' +"Day_" + str(dayNum+5) + "_Hour_" + str(j + 3) + ".pdf")
        plt.clf()
        print "Done!"

def plotweather(submitfile, weatherfile,cityLocFile,hourNum = 18,xsize = 548,ysize = 421,maxDay = 5,maxCity = 10,threshold = 15):
    cityloc = pd.read_csv(cityLocFile)
    xCity = cityloc['xid']
    yCity = cityloc['yid']
    x = np.linspace(1, xsize, xsize)
    y = np.linspace(1, ysize, ysize)
    X,Y = np.meshgrid(y, x)
    if submitfile != None:
        pathfile = pd.read_csv(submitfile,header = None, names = ["city","Day","Time","x","y"])        
    for dayNum in range(1,maxDay+1):
        windGraph,rainGraph = get_Wind_Rain_Graph(weatherfile,dayNum)
        for city in range(1,maxCity+1):
            if submitfile == None:
                plot_func_ref(dayNum,windGraph,rainGraph,xCity,yCity,hourNum,X,Y)
                break
            else:
                pathtmp = getpathtmp(pathfile,city,dayNum)
                Len = pathtmp.shape[0]
                if Len > 0: 
                    pathpiece,initial = pathtmp[["x","y"]],pathtmp["Time"][0]
                    plot_func(dayNum,city,pathpiece,windGraph,rainGraph,xCity,yCity,hourNum,partition(Len,initial),X,Y)
                    
                    
                    
                    
                    
    

