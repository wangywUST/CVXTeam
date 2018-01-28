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
        if data[temp_x, temp_y] < threshold and i >= (height) * 30 :
            index = i
            New_start_pos = L[i]
            Stop = True
        i = i + 1
    if Stop:
        mid = (index + len(L)) // 2 
        L = L[0:index] + L[index] * (len(L)-index)
    return New_start_pos, L
    