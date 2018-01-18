# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:33:41 2018

@author: lzhaoai
"""

import pandas as pd
import sys
import numpy as np
sys.path.append("Functions/Linlong")
sys.path.append("Functions/Linlong/dijkstar")
from Module3_func import *


submitfile = "C:\Users\lwuag\Desktop\GitHub-Project\CVXTeam\Data\submitResult_Linlong.csv"
weatherfile = "C:\Users\lwuag\Desktop\TianchiData\predict_model_2.csv"
cityLocFile = "C:\Users\lwuag\Desktop\TianchiData\CityData.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
plotweather(submitfile, weatherfile,cityLocFile)