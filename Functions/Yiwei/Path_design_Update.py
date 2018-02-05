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
from New_start_point_backtracking import *
from Remedy_4_no_way import *
from check_start_point import *

def Path_design_Update(Data, star_point, end_point, end_point_replace, height, threshold):
    high_num = int(Data.shape[0])
    row_num = int(Data.shape[1])
    col_num = int(Data.shape[2])      
    end_x = end_point // col_num
    end_y = end_point % col_num
    star_x = star_point // col_num
    star_y = star_point % col_num
#%% check the feasibility of the end point
    if Data[height,end_x,end_y] >= threshold:
        size_bound = 60
        print('new end point')
        end_point_replace = New_end_point_nearby(Data[height,:,:], star_point, end_point, col_num, size_bound, threshold) 
    else:
        end_point_replace = end_point
#    print(end_point_replace)
#    print(end_point)
#    print(end_point_replace // col_num)
#    print(end_point_replace % col_num)
#    if Data[height, end_x, end_y] >= 1:
#        end_point_replace = New_end_point_mid(Data[height,:,:], star_point, end_point, col_num)       
#    else:
#        end_point_replace = end_point

#%% check the feasibility of the start point   
#    input: existing path 
#    output: new start_point
    
    star_x = star_point // col_num
    star_y = star_point % col_num 
    if Data[height, star_x, star_y] >= threshold:
        return []
#    if height != 0 and Data[height, star_x, star_y] >= threshold:
#        star_point, Trace= New_start_point_backtracking(Data[height,:,:], Trace, col_num, star_point, threshold, height)
        
#%% geenrate the graph   
    graph = Graph()
    tune_para = 0
    for i in range(row_num):
        for j in range(col_num):            
#            graph.add_node(index)
            if Data[height, i, j] < threshold:
                index = i * col_num + j
                if i - 1 >= 0 and Data[height,i - 1, j] < threshold:
                    cost = 2 + tune_para * Data[height,i - 1, j] * 1.0 / 15 * 2
                    index_next = (i - 1) * col_num + j
                    graph.add_edge(index, index_next, {'cost': cost})
                if i + 1 < row_num and Data[height, i + 1, j] < threshold:
                    cost = 2 + tune_para * Data[height,i + 1, j] * 1.0 / 15 * 2
                    index_next = (i + 1) * col_num + j
                    graph.add_edge(index, index_next, {'cost': cost})
                if j - 1 >= 0 and Data[height, i, j - 1] < threshold:
                    cost = 2 + tune_para * Data[height, i, j - 1] * 1.0 / 15 * 2
                    index_next = i * col_num + (j - 1)
                    graph.add_edge(index, index_next, {'cost': cost})
                if j + 1 < col_num and Data[height, i, j + 1] < threshold:
                    cost = 2 + tune_para * Data[height, i, j + 1] * 1.0 / 15 * 2
                    index_next = i * col_num + (j + 1)
                    graph.add_edge(index, index_next, {'cost': cost})
    cost_func_1 = lambda u, v, e, prev_e: e['cost']
    heuristic_func_1 = lambda u, v, e, prev_e: e['cost']    
#    cost_func_1 = None
#    heuristic_func_1 = None
#    PathInfo = find_path(graph, star_point, end_point_replace, cost_func=cost_func_1, heuristic_func=heuristic_func_1)
    try:
        PathInfo = find_path(graph, star_point, end_point_replace, cost_func=cost_func_1, heuristic_func=heuristic_func_1)
        PathInfo = PathInfo.nodes  
#        print(PathInfo)
    except:
#        print(1111)
        PathInfo = Remedy_4_no_way(Data, star_point, end_point) # stand still
    
#    PathInfo = PathInfo.nodes    
#%%
    if height == high_num - 1:
        print('A')
        return PathInfo
    else:
        if PathInfo[-1] == end_point:
            if len(PathInfo) <= 30:
                print('B')
                return PathInfo
            else:
                Stop = False
                Height_pos = 0
                index = 0
                while index in range(0, len(PathInfo)) and not Stop:
                    z_id = min(height + index // 30, 17)
                    x_id = PathInfo[index] // col_num
                    y_id = PathInfo[index] % col_num
                    if Data[z_id, x_id, y_id] >= threshold:
                        Stop = True
                        Height_pos = z_id
                    index = index + 1
                if Stop:                   
                    start_index = (Height_pos - height)*30-1
                    start_point = PathInfo[start_index]
                    start_replace, PathInfo = check_start_point(PathInfo[0:start_index], start_point, start_index, Height_pos, Data, threshold)
#                    Height_pos = start_replace // 30
                    print('C')
                    return PathInfo + Path_design_Update(Data, start_replace, end_point, end_point, Height_pos, threshold)
                else:
                    print('D')
                    return PathInfo
        elif PathInfo[-1] == end_point_replace:
            if len(PathInfo) < 30:
                print('E')
                return PathInfo + [end_point_replace] * (29 - len(PathInfo)) + Path_design_Update(Data, end_point_replace, end_point, end_point, height + 1, threshold)
            elif len(PathInfo) == 30:
                print('F')
                return PathInfo[0:(len(PathInfo)-1)]  + Path_design_Update(Data, end_point_replace, end_point, end_point, height + 1, threshold)
            else:
                Stop = False
                Height_pos = 0
                index = 0
                while index in range(0, len(PathInfo)) and not Stop:
                    z_id = min(height + index // 30, 17)
                    x_id = PathInfo[index] // col_num
                    y_id = PathInfo[index] % col_num
                    if Data[z_id, x_id, y_id] >= threshold:
                        Stop = True
                        Height_pos = z_id
                    index = index + 1
                if Stop:                   
                    start_index = (Height_pos - height)*30-1
                    start_point = PathInfo[start_index]
                    start_replace, PathInfo = check_start_point(PathInfo[0:start_index], start_point, start_index, Height_pos, Data, threshold) 
#                    Height_pos = start_replace // 30
                    print('G')
                    return PathInfo  + Path_design_Update(Data, start_replace, end_point, end_point, Height_pos, threshold)
                else:
                    print('H')
                    return PathInfo
        else:
#            Stop = False
#            Height_pos = 0
#            index = 0
#            while index in range(0, len(PathInfo)) and not Stop:
#                z_id = height + index // 30
#                x_id = PathInfo[index] // col_num
#                y_id = PathInfo[index] % col_num
#                if Data[z_id, x_id, y_id] >= threshold:
#                    Stop = True
#                    Height_pos = z_id
#                index = index + 1
#            if Stop:
#                return PathInfo[0:((Height_pos - height)*30-1)] + Path_design_Update(Data, PathInfo[(Height_pos - height)*30-1], end_point, end_point, Height_pos, threshold)
#            else:
#                return PathInfo
            print('I')
            return PathInfo[0:(len(PathInfo)-1)] + Path_design_Update(Data, PathInfo[-1], end_point, end_point, height + 1, threshold)
        
#%%
#    Stop = False
#    Height_pos = 0
#    if height == high_num - 1 or PathInfo == []:
#        return PathInfo.nodes
#    else:
#        while index in range(0, len(PathInfo.nodes)) and not Stop:
#            z_id = index // 30
#            x_id = PathInfo[index].nodes // col_num
#            y_id = PathInfo[index].nodes % col_num
#            if Data[z_id, x_id, y_id] >= 1:
#                Stop = True
#                Height_pos = z_id
#        if Stop:
#            end_point_replace = end_point
#            return PathInfo[0:Height_pos*30].nodes + Path_design_Update(Data, PathInfo[Height_pos*30-1].nodes, end_point, end_point_replace, Height_pos, threshold)
#        else:
#            return PathInfo.nodes
        
