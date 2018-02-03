# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 12:24:41 2018

@author: Yiwei
"""
#%% Path Setting --------------------------------------------------------------------------
#Input Paths
submitfile = "Data/submitResult_Yiwei_20180131.csv"
weatherfile = "/home/ust.hk/ywanggp/tsclient/ywanggp/Dropbox/Contest/contest/Data/predict_model_3.csv"
cityLocFile = "Data/CityData.csv"

#Function Paths
funLib1 = "Functions/Yiwei"
funLib2 = "Functions/Yiwei/dijkstar"


#%% Parameter Setting --------------------------------------------------------------------
windThres = 15


#%% Main Function ------------------------------------------------------------------------
import sys
sys.path.append(funLib1)
sys.path.append(funLib2)    
from Module3_func import *
import pandas as pd
import numpy as np

Score = obtainScore(submitfile, weatherfile, cityLocFile,threshold = windThres)