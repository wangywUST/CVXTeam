# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:38:47 2018

@author: lwuag
"""

def New_start_point_backtracking(data, L, col_num, star_point, threshold, height):
    Stop = False
    index = None
    New_start_pos = star_point
    i = 0
    while i in range(len(L)) and not Stop:
        temp_x = L[i] // col_num
        temp_y = L[i] % col_num
        if data[temp_x, temp_y] < threshold:
            index = i
            New_start_pos = L[i]
            Stop = True
        i = i + 1
    if Stop:
        mid = (index + len(L)) // 2 
        L = L[0:index] + L[index:mid] + L[mid:(index-1):-1]
    return Stop, New_start_pos
    L = L[::-1]
    i = 0
    New_start_pos = star_point
    while i in range(len(L)) and not Stop:
        temp_x = L[i] // col_num
        temp_y = L[i] % col_num
        if data[temp_x, temp_y] < threshold:
            Stop = True
            New_start_pos = L[i]
    return New_start_pos
    