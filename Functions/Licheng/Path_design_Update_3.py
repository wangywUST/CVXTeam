# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 15:07:38 2018

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
from check_End_as_Start import *

def  Path_design_Update_3(Data, star_point, true_end, end_point_replace, startHours, startMins, height, threshold):
    high_num = int(Data.shape[0])
    row_num = int(Data.shape[1])
    col_num = int(Data.shape[2])      
    end_x = end_point_replace // col_num
    end_y = end_point_replace % col_num
    star_x = star_point // col_num
    star_y = star_point % col_num
#%% check the feasibility of the end point
    if Data[height,end_x,end_y] >= threshold:
        size_bound = 60
        print('generating new end...')
        end_point_replace = New_end_point_nearby(Data[height, :, :], star_point, end_point_replace, col_num, size_bound, threshold) 
#    else:
#        end_point_replace = end_point
        
    

#%% check the feasibility of the start point     
    star_x = star_point // col_num
    star_y = star_point % col_num 
    if Data[height, star_x, star_y] >= threshold:
        print('StartDead')
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
                    cost = 2 + tune_para * Data[height,i - 1, j] * 1.0 / 30 * 2
                    index_next = (i - 1) * col_num + j
                    graph.add_edge(index, index_next, {'cost': cost})
                if i + 1 < row_num and Data[height, i + 1, j] < threshold:
                    cost = 2 + tune_para * Data[height,i + 1, j] * 1.0 / 30 * 2
                    index_next = (i + 1) * col_num + j
                    graph.add_edge(index, index_next, {'cost': cost})
                if j - 1 >= 0 and Data[height, i, j - 1] < threshold:
                    cost = 2 + tune_para * Data[height, i, j - 1] * 1.0 / 30 * 2
                    index_next = i * col_num + (j - 1)
                    graph.add_edge(index, index_next, {'cost': cost})
                if j + 1 < col_num and Data[height, i, j + 1] < threshold:
                    cost = 2 + tune_para * Data[height, i, j + 1] * 1.0 / 30 * 2
                    index_next = i * col_num + (j + 1)
                    graph.add_edge(index, index_next, {'cost': cost})
    cost_func_1 = lambda u, v, e, prev_e: e['cost']
    heuristic_func_1 = lambda u, v, e, prev_e: e['cost']    
#    cost_func_1 = None
#    heuristic_func_1 = None
#    PathInfo = find_path(graph, star_point, end_point_replace, cost_func=cost_func_1, heuristic_func=heuristic_func_1)
    try:
        print("Generating the path...")
        PathInfo = find_path(graph, star_point, end_point_replace, cost_func=cost_func_1, heuristic_func=heuristic_func_1)
        PathInfo = PathInfo.nodes  
#        print(PathInfo)
    except:
        print("No way for DJ, stand still for one hour")
        PathInfo = Remedy_4_no_way(Data, star_point, true_end) # stand still
    
#    PathInfo = PathInfo.nodes    
#%%
    if height == startHours:
        stp_nums = 30 - startMins // 4 
        if height == high_num - 1:
            print('first stage: TimeOut')
            return PathInfo
        else:
            if Data[height + 1, PathInfo[-1] // col_num, PathInfo[-1] % col_num] < threshold:
                if PathInfo[-1] == true_end:
                    if len(PathInfo) <= stp_nums:
                        print('first stage: Good')
                        return PathInfo
                    else:
                        print("first stage: end is true end but cannot reach in one hour")
                        return PathInfo[0:stp_nums] + Path_design_Update_3(Data, PathInfo[stp_nums], true_end, true_end, startHours, startMins, height+1, threshold)
                elif PathInfo[-1] == end_point_replace:
                    if len(PathInfo) <= stp_nums:
                        print('first stage: end is tempororary, reach the point')
                        return PathInfo + [end_point_replace] * (stp_nums - len(PathInfo)) + Path_design_Update_3(Data, PathInfo[-1], true_end, true_end, startHours, startMins, height + 1, threshold)
                    else:
                        print('first stage: end is tempororary, cannot reach the point')
                        return PathInfo[0:stp_nums] + Path_design_Update_3(Data, PathInfo[stp_nums], true_end, true_end, startHours, startMins, height+1, threshold)
                else:
                    print('first stage: end is neither ture nor temporary')
                    return PathInfo[0:stp_nums] + Path_design_Update_3(Data, PathInfo[-1], true_end, true_end, startHours, startMins, height + 1, threshold)
            else:
                if len(PathInfo) < stp_nums:
                    print('first stage: arrived but end is deadzone, generate new and rerun for the same hour')
                    size_bound = 60
                    end_point_replace = check_End_as_Start(Data[height + 1, :, :], star_point, PathInfo[-1], col_num, size_bound, threshold)
                    return Path_design_Update_3(Data, star_point, true_end, end_point_replace, startHours, startMins, height, threshold)
                else:
                    print('first stage: not arrived but end is deadzone, go along this way')
                    return PathInfo[0:stp_nums] + Path_design_Update_3(Data, PathInfo[stp_nums], true_end, true_end, startHours, startMins, height+1, threshold)
    else:       
        if height == high_num - 1:
            print('new stage: TimeOut')
            return PathInfo
        else:
            if Data[height + 1, PathInfo[-1] // col_num, PathInfo[-1] % col_num] < threshold:
                if PathInfo[-1] == true_end:
                    if len(PathInfo) <= 30:
                        print('new stage: Good')
                        return PathInfo
                    else:
                        print("new stage: end is true end but cannot reach in one hour")
                        return PathInfo[0:30] + Path_design_Update_3(Data, PathInfo[30], true_end, true_end, startHours, startMins, height+1, threshold)
                elif PathInfo[-1] == end_point_replace:
                    if len(PathInfo) <= 30:
                        print('new stage: end is tempororary, reach the point')
                        return PathInfo + [end_point_replace] * (30 - len(PathInfo)) + Path_design_Update_3(Data, PathInfo[-1], true_end, true_end, startHours, startMins, height + 1, threshold)
                    else:
                        print('new stage: end is tempororary, cannot reach the point')
                        return PathInfo[0:30] + Path_design_Update_3(Data, PathInfo[30], true_end, true_end, startHours, startMins, height+1, threshold)
                else:
                    print('new stage: arrived but end is deadzone, generate new and rerun for the same hour')
                    return PathInfo[0:30] + Path_design_Update_3(Data, PathInfo[-1], true_end, true_end, startHours, startMins, height + 1, threshold)
            else:
                if len(PathInfo) < 30:
                    print('new stage: arrived but end is deadzone, generate new and rerun for the same hour')
                    size_bound = 60
                    end_point_replace = check_End_as_Start(Data[height + 1, :, :], star_point, PathInfo[-1], col_num, size_bound, threshold)
                    return Path_design_Update_3(Data, star_point, true_end, end_point_replace, startHours, startMins, height, threshold)
                else:
                    print('new stage: not arrived but end is deadzone, go along this way')
                    return PathInfo[0:30] + Path_design_Update_3(Data, PathInfo[30], true_end, true_end, startHours, startMins, height+1, threshold)

