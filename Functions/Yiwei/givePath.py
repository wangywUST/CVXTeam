# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:37 2018

@author: ywanggp
"""
from heuristicSquareAlg import *
from showPoint import *
import numpy as np
from fillPath import *
from obtainScore import *

def givePath(windGraph, Departure, destination, xsize, ysize, xCity, yCity):
    iniLoc = Departure.copy()
    newLoc = iniLoc.copy()
    hourNum = 18
    timePieceMin = 60
    locList = newLoc 
    for i in range(hourNum):
        lastLoc = newLoc.copy()
        windGra = windGraph[i,:,:]
        (newLoc, flag) = heuristicSquareAlg(windGra, lastLoc.copy(), destination, timePieceMin)
#        print(windGra[newLoc[0], newLoc[1]])
        Fillpath = fillPath(lastLoc, newLoc)
        if(i == 0):
            locList = np.concatenate((np.asarray([locList]), Fillpath), axis = 0)
        else:
            locList = np.concatenate((locList, Fillpath), axis = 0)        
        showPoint(windGra, xCity, yCity, locList)
        if(flag == True):
            break    
    return(locList,obtainScore(locList,windGraph))