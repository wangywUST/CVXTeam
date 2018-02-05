# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:33:41 2018

@author: lzhaoai
"""

import sys
sys.path.append("Functions/Linlong")
from Module3_func import *


submitfile = "C:\Users\lwuag\Desktop\GitHub-Project\CVXTeam\Data\submitResult_Linlong_20180205.csv"
weatherfile = "C:\Users\lwuag\Dropbox\With Licheng\contest\Input\predict_model_2.csv"
cityLocFile = "C:\Users\lwuag\Desktop\GitHub-Project\CVXTeam\Data\CityData.csv"

#Score = obtainScore(submitfile, weatherfile,cityLocFile)
plotweather(submitfile, weatherfile,cityLocFile)
obtainScore(submitfile, weatherfile,cityLocFile)