# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:33:08 2018

@author: lwuag
"""
#from Path_design_Update import *
from Path_design_Update_1 import *
from Path_design_Update_2 import *
#from Remedy_4_no_way import *
#from Remedy_4_false_end import *
from index_2_xy import *
from Data_convert import *
import numpy as np

def Path_generator(windGraph, xCity_begin, yCity_begin, xCity_end, yCity_end, thre_wind, height):
    
    ysize = int(windGraph.shape[2])
    star_point = xCity_begin * ysize + yCity_begin
    end_point = xCity_end * ysize + yCity_end
    Data = Data_convert(windGraph, thre_wind, MultiLay=False, Convert = False)
    threshold = 15
#    Pathinfo = Path_design(Data, star_point, end_point, end_point, height)
#    Pathinfo = Path_design_Update(Data, star_point, end_point, end_point, height, threshold)
    path_temp = []
    Pathinfo = Path_design_Update_2(Data, path_temp, star_point, end_point, end_point, height, threshold)
        
#    end_pos = Pathinfo[-1]
#    end_x, end_y = index_2_xy(end_pos, Data.shape[2])
#    if end_x != xCity_end or end_y != yCity_end:
#        if len(Pathinfo) <= 540:
#            Pathinfo = []
#        else:
#            Pathinfo = Remedy_4_false_end(Data, Pathinfo, star_point, end_point)
     
    Pathinfo = np.asarray([[node/ysize, node%ysize] for node in Pathinfo])
    return Pathinfo