# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:19:30 2018

@author: lwuag
"""
from index_2_xy import *
import numpy as np

#def spiral(X, Y):
#    x = y = 0
#    dx = 0
#    dy = -1
#    L = []
#    for i in range(max(X, Y)**2):
#        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
#            L = L + [x, y]
#        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
#            dx, dy = -dy, dx
#        x, y = x+dx, y+dy
#    num = len(L)
#    L = np.asarray(L)
#    L = L.reshape((num//2, 2))
#    return L
#
#def check_End_as_Start(data, star_point, end_point, col_num, size_bound, threshold):
#    end_x, end_y = index_2_xy(end_point, col_num)
#    Stop = False
#    size = 1
#    New_end_point = star_point
#    start_pos = 0
#    while not Stop and size <= size_bound:  
#        Move = spiral(size, size)
#        i = 0
#        while i in range(start_pos, int(Move.shape[0])) and not Stop:
#            New_endx = end_x + Move[i, 0]
#            New_endy = end_y + Move[i, 1]
#            if data[New_endx, New_endy] < threshold:
#                Stop = True
#                New_end_point = New_endx * col_num + New_endy
#            i = i + 1       
#        start_pos = Move.shape[0]
#        size = size + 1        
#    return New_end_point

def check_End_as_Start(data, path_temp, PathInfo, star_point, end_point, col_num, size_bound, threshold):
    temp = path_temp + PathInfo
    temp = temp[::-1]
    i = 0
    stop_1 = False
    stop_2 = False
    while i in range(len(PathInfo)) and not stop_1:
        if data[PathInfo[len(PathInfo)-1-i]//col_num,PathInfo[len(PathInfo)-1-i]%col_num] < threshold:
            index_1 = len(PathInfo)-1-i
            stop_1 = True
        i = i + 1
    j = 0 
    while j in range(len(temp)) and not stop_2:
        if data[path_temp[len(path_temp)-1-j]//col_num,path_temp[len(path_temp)-1-j]%col_num] < threshold:
            index_2 = len(path_temp)-1-j
            stop_2 = True
        j = j + 1
    if stop_1:
        path_temp = path_temp + PathInfo[0:index_1] + [PathInfo[index_1]] * (30-index_1)
        return path_temp
    else:
        if stop_2:
            print(index_2)
            path_temp = path_temp[0:index_2] + [path_temp[index_2]]*(len(path_temp)-index_2)
            return path_temp
        else:
            path_temp = path_temp + PathInfo
            return path_temp
    