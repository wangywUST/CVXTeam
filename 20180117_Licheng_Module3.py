# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:33:41 2018

@author: lzhaoai
"""

import pandas as pd
import sys
import numpy as np
sys.path.append("Functions/Licheng")
sys.path.append("Functions/Licheng/dijkstar")
from Module3_func import *


submitfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\CompareResults\submitResult_0116.csv"
weatherfile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"

#inputfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\submitResult_Licheng.csv"
#outputfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\CompareResults\submitResult_0117.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
plotweather(submitfile, weatherfile,cityLocFile)
#cityfilter(inputfile,outputfile)