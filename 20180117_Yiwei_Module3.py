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


submitfile = "C:\Users\lzhaoai\Desktop\Tianchi\CVXTeam\Data\submitResult.csv"
weatherfile = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_2.csv"
cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
plotweather(submitfile, weatherfile,cityLocFile)