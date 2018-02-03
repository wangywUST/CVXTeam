# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:50:33 2018

@author: lwuag
"""

def index_2_xy(index, col_num):
    x_id = index // col_num
    y_id = index % col_num
    return x_id, y_id