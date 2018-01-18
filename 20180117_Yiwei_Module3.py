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


submitfile = "Data/CompareResults/submitResult_0115.csv"
weatherfile = "C:\Users\wangyw\Dropbox\Contest\contest\Data\predict_model_2.csv"
cityLocFile = "Data\CityData.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
print obtainScore(submitfile, weatherfile, cityLocFile)
#plotweather(submitfile, weatherfile,cityLocFile)