# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:17:13 2018

@author: lwuag
"""

def Data_convert(Data, thre_wind):  
    for k in range(Data.shape[0]):
        for i in range(Data.shape[1]):
            for j in range(Data.shape[2]):
                    Data[k, i, j] = 0 if Data[k, i, j] < thre_wind else 1
#    for k in range(Data.shape[0]-1):
#        Data[k,:,:] = Data[k, :, :] + Data[k + 1, :, :]
#    for k in range(Data.shape[0]):
#        for i in range(Data.shape[1]):
#            for j in range(Data.shape[2]):
#                    Data[k, i, j] = 0 if Data[k, i, j] < 1 else 1
    return Data