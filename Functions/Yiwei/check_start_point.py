# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:34:15 2018

@author: lwuag
"""

def check_start_point(PathInfo, start_point, start_index, Height_pos, Data, threshold):
    col_num = int(Data.shape[2]) 
    start_x = start_point // col_num
    start_y = start_point % col_num
    start_replace = start_point
    start_replace_index = start_index
    Stop = False
    if Data[Height_pos, start_x, start_y] >= threshold:
        Stop = True
    Temp = PathInfo[::-1]
    if Stop:
        i = 0
        Flag = False
        while i in range(min(30, len(Temp))) and not Flag:
            print(i)
            if Data[Height_pos, Temp[i]//col_num, Temp[i]%col_num] < threshold:
                Flag = True
                start_replace = Temp[i]
                start_replace_index = len(PathInfo) - 1 - i
            i = i + 1
        if Flag:
            PathInfo = PathInfo[0: start_replace_index] + [PathInfo[start_replace_index]] * (len(PathInfo) - start_replace_index)
    return start_replace, PathInfo
        
        
    