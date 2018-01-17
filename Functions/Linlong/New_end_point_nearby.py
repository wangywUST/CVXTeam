# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:29:10 2018

@author: lwuag
"""

from index_2_xy import *
from xy_2_index import *
import numpy as np

def spiral(X, Y):
    x = y = 0
    dx = 0
    dy = -1
    L = []
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            L = L + [x, y]
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    num = len(L)
    L = np.asarray(L)
    L = L.reshape((num//2, 2))
    return L

def New_end_point_nearby(data, star_point, end_point, col_num, size):
    start_x, start_y = index_2_xy(star_point, col_num)
    end_x, end_y = index_2_xy(end_point, col_num)
    Stop = False
    while not Stop:    
        Move = spiral(size, size)
        i = 0
        while i in range(int(Move.shape[0])) and not Stop:
            New_endx = end_x + Move[i, 0]
            New_endy = end_y + Move[i, 1]
            if data[New_endx, New_endy] < 1:
                Stop = True
                New_end_point = New_endx * col_num + New_endy
            i = i + 1
        size = size + 1
    return New_end_point