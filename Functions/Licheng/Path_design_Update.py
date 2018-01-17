# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:04:05 2018

@author: lwuag
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:43:42 2018

@author: lwuag
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:25:24 2018

@author: lwuag
"""
import numpy as np
from graph import *
from algorithm import *
#from New_end_point_mid import *
from New_end_point_nearby import *


#    star_point = Start_city[0] * col_num + Start_city[1]
#    end_point = Target_city[0] * col_num + Target_city[1]

def Path_design_Update(Data, star_point, end_point, end_point_replace, height):
    high_num = int(Data.shape[0])
    row_num = int(Data.shape[1])
    col_num = int(Data.shape[2])      
    end_x = end_point // col_num
    end_y = end_point % col_num
    
#%%   
    size_init = 5
    end_point_replace = New_end_point_nearby(Data[height,:,:], star_point, end_point, col_num, size_init) 
#    if Data[height, end_x, end_y] >= 1:
#        end_point_replace = New_end_point_mid(Data[height,:,:], star_point, end_point, col_num)       
#    else:
#        end_point_replace = end_point
    
#%%        
    graph = Graph()
    for i in range(row_num):
        for j in range(col_num):
            index = i * col_num + j
#            graph.add_node(index)
            if i - 1 >= 0 and Data[height,i - 1, j] < 1:
                index_next = (i - 1) * col_num + j
                graph.add_edge(index, index_next, {'cost': 2})
            if i + 1 < row_num and Data[height, i + 1, j] < 1:
                index_next = (i + 1) * col_num + j
                graph.add_edge(index, index_next, {'cost': 2})
            if j - 1 >= 0 and Data[height, i, j - 1] < 1:
                index_next = i * col_num + (j - 1)
                graph.add_edge(index, index_next, {'cost': 2})
            if j + 1 < col_num and Data[height, i, j + 1] < 1:
                index_next = i * col_num + (j + 1)
                graph.add_edge(index, index_next, {'cost': 2})
    cost_func_1 = lambda u, v, e, prev_e: e['cost']
    heuristic_func_1 = lambda u, v, e, prev_e: e['cost']
    PathInfo = find_path(graph, star_point, end_point_replace, cost_func=cost_func_1, heuristic_func=heuristic_func_1)
    
#%%
    Stop = False
#    Fail_pos = 0
    Height_pos = 0
    if height == high_num - 1:
        return PathInfo.nodes
    else:
        while index in range(0, len(PathInfo.nodes)) and not Stop:
            z_id = index // 30
            x_id = PathInfo[index].nodes // col_num
            y_id = PathInfo[index].nodes % col_num
            if Data[z_id, x_id, y_id] >= 1:
                Stop = True
#                Fail_pos = index
                Height_pos = z_id
        if Stop:
            end_point_replace = end_point
            return PathInfo[0:(Height_pos*30-1)].nodes + Path_design(Data, PathInfo[Height_pos*30-1].nodes, end_point, end_point_replace, Height_pos)
        else:
            return PathInfo.nodes
        
