# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:49:31 2018

@author: lwuag
"""
from index_2_xy import *
from xy_2_index import *

def New_end_point_mid(data, star_point, end_point, col_num):
    start_x, start_y = index_2_xy(star_point, col_num)
    end_x, end_y = index_2_xy(end_point, col_num)
    Stop = False
    while not Stop:
        mid_pos_x = (start_x + end_x) // 2
        mid_pos_y = (start_y + end_y) // 2
        if data[mid_pos_x, mid_pos_y] < 1:
            Stop = True
        else:
            end_x = mid_pos_x
            end_y = mid_pos_y
    end_point_replace = xy_2_index(mid_pos_x, mid_pos_y, col_num)
    return end_point_replace