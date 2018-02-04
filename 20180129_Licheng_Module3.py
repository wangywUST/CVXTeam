# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:33:41 2018

@author: lzhaoai
"""


import sys
sys.path.append("Functions/Licheng")
from Module3_func import *

#submitfile = "C:\Users\lzhaoai\Dropbox\With Licheng and Linlong\contest\submit20180116\submitResult_20180119_2.csv"
submitfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\submitResult_Licheng_20180204.csv"
weatherfile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"
#submitfile = None

#Score = obtainScore(submitfile, weatherfile,cityLocFile,threshold = 15)
plotweather(submitfile, weatherfile,cityLocFile)