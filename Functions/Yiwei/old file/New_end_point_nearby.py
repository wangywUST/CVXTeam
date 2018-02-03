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
        
def New_end_point_nearby(data, star_point, end_point, col_num, size, threshold):
    start_x, start_y = index_2_xy(star_point, col_num)
    end_x, end_y = index_2_xy(end_point, col_num)
    Stop = False
    while not Stop:    
        Move = spiral(size, size)
        i = 0
        while i in range(int(Move.shape[0])) and not Stop:
            New_endx = end_x + Move[i, 0]
            New_endy = end_y + Move[i, 1]
            if data[New_endx, New_endy] < threshold:
                Stop = True
                New_end_point = New_endx * col_num + New_endy
            i = i + 1
        size = size + 1
    return New_end_point


#def ring(jump):
#    if(jump == 0):
#        return [[0,0]]
#    else:
#        L = []
#        for i in range(2*jump):
#            L += [[-jump+i,-jump]]
#        for i in range(2*jump):
#            L += [[jump,-jump+i]]
#        for i in range(2*jump):
#            L += [[jump-i,jump]]
#        for i in range(2*jump):
#            L += [[-jump,jump-i]]
#        return L
#
#def filterfeasible(Data,L,end_point,col_num,thres):
#    end_x, end_y = index_2_xy(end_point, col_num)
#    return [[end_x+x[0],end_y+x[1]] for x in L if Data[end_x+x[0],end_y+x[1]] < thres]
#    
#            
#        
#
#def New_end_point_nearby(data, star_point, end_point, col_num, size, threshold = 15):
#    start_x, start_y = index_2_xy(star_point, col_num)
#    end_x, end_y = index_2_xy(end_point, col_num)
#    Stop = False
#    s = 0
#    New_end_point = end_point
#    while s < size:   
#        L = filterfeasible(data,ring(s),end_point,col_num,threshold)
#        if len(L) > 0:
#            dist = [abs(x[0] - start_x) + abs(x[1] - start_y) for x in L]
#            for i in range(len(dist)):
#                if dist[i] == min(dist):
#                    New_end_point = np.asarray(L[i])
#                    return New_end_point
#        else:
#            s += 1 
#    return New_end_point