# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:17:13 2018

@author: lwuag
"""
import numpy as np
def Data_convert(Data, thre_wind, MultiLay=False, Convert = False):  
    if Convert:
        Data_ = np.zeros(Data.shape)
        Data_[Data>=thre_wind] = 1
        if MultiLay:
            Data_ = Data_[:Data.shape[0]-1,:,:] + Data_[1:,:,:]
            Data_[Data_>=1] = 1 
        Data_ = Data_.astype(int)
    else:
        Data_ = Data.copy()
    return Data_