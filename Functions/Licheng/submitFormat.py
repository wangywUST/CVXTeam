# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def submitFormat(dayNum, cityNum, pathList):
    des_n_day = np.array([[cityNum,dayNum] for _ in range(pathList.shape[0])])
    time = [3,0]
    string = ["03:00"]
    for _ in range(pathList.shape[0]-1):
        time = [time[0],time[1]+2] if time[1]<58 else [time[0]+1,0]
        string_sub = ""
        if time[0] < 10:
            string_sub += "0"+str(time[0])
        else:
            string_sub += str(time[0])
        string_sub += ":"
        if time[1] < 10:
            string_sub += "0"+str(time[1])
        else:
            string_sub += str(time[1]) 
        string += [string_sub] 
    string = np.asarray(string).reshape(-1,1)
    return string.copy(), des_n_day.copy()