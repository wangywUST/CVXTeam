#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:32:01 2018

@author: ywanggp
"""
#%% Setting Paths  ------------------------------------------------------------------------------
#Input Paths
trainPredFile = "/home/ust.hk/ywanggp/tsclient/ywanggp/Downloads/ForecastDataforTraining_201712.csv"
trainTrueFile = "/home/ust.hk/ywanggp/tsclient/ywanggp/Downloads/In_situMeasurementforTraining_201712.csv"
testPredFile = "/home/ust.hk/ywanggp/tsclient/ywanggp/Downloads/ForecastDataforTesting_201712.csv"
cityLocFile = "Data/CityData.csv"
testTrueFile = "/home/ust.hk/ywanggp/tsclient/ywanggp/Dropbox/Contest/contest/Data/predict_model_3.csv"

#Output Paths
submitPath = "Data/submitResult_Yiwei_20180131.csv"

#Function Paths
LichengFunction = "Functions/Licheng"
LichengFunctionSub1 = "Functions/Licheng/dijkstar"

#%% Setting Global Parameters ---------------------------------------------------------------------
#Map Size
xsize = 548
ysize = 421

#Number of Days, Citys, and Hours
maxDay = 5
maxCity = 10
hourNum = 18

#The threshold of the dangerous wind speed
thre_wind = 15

#Executing Ranges
dayList = [1] #The days that would be dealt with (1 - 5)
cityList = [1] #The citys that would be dealt with (1 - 10)

#One map's data size
chunksize = xsize * ysize


#%% Defining Functions -----------------------------------------------------------------------------
import sys
sys.path.append("Functions/Licheng")
sys.path.append("Functions/Licheng/dijkstar")

from jumpDays import *
import numpy as np
import pandas as pd

#Get wind graph of one day. 
#Output: Return a 3 dimensional matrix containing 18 layers (18 hours), 
#one xsize by ysize double matrix per layer.
#Input: index of the day, int, [1, 5]
def getWindGraph(dayIndex):
    windGraph = np.zeros((hourNum,xsize,ysize))
    df = pd.read_csv(testTrueFile, chunksize = chunksize)
    df = jumpDays(df, dayIndex-1, chunksize)
    for _ in range(hourNum):
        windGra = df.get_chunk(chunksize)["wind"]
        windGraph[_,:,:] = windGra.values.reshape(xsize,ysize).copy()
    return windGraph


from submitFormat import *
from Path_generator import *
#Add new Paths of one city, one day to the existing block.
#Output: Return extended part.
#Input: existing block.
def extendBlock():
    star_point = xCity[0] * ysize + yCity[0]
    extendedPart = []
    for cityNum in cityList:
        height = 0
        Pathinfo = Path_generator(windGraph, xCity[0], yCity[0], xCity[cityNum], yCity[cityNum], thre_wind, height)     
        (des_n_city, des_n_day) = submitFormat(dayNum+5, cityNum, Pathinfo)
        extendedPart += list(np.concatenate((des_n_day, des_n_city, Pathinfo), axis = 1))
    return extendedPart

#Write the block containing path information to the output path
#Input: existing block.
def writeToSubmitFile(block):
    block = np.asarray(block)
    df_b = pd.DataFrame(block)
    df_b.to_csv(submitPath, header=None,index = False)
    




#%% Main Function ----------------------------------------------------------------------------'
if __name__ == "__main__":
    cityLoc = pd.read_csv(cityLocFile)       #City Location List, 11 By 3 Dataframe: Index, x, y
    xCity = cityLoc['xid']                   #x location list of cities
    yCity = cityLoc['yid']                   #y location list of cities
    
    block = []  #containing result path information 
    for dayNum in dayList:
        windGraph = getWindGraph(dayNum)
        block += extendBlock()
    writeToSubmitFile(block)

