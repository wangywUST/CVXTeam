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
import matplotlib.pyplot as plt


submitfile = "Data/CompareResults/submitResultBest_0116.csv"
weatherfile = "C:\Users\wangyw\Dropbox\Contest\contest\Data\predict_model_2.csv"
cityLocFile = "Data\CityData.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
plt.plot(obtainScore(submitfile, weatherfile, cityLocFile))
print sum(obtainScore(submitfile, weatherfile, cityLocFile))
plt.show()
#plotweather(submitfile, weatherfile,cityLocFile)