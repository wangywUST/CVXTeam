# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:21:17 2018

@author: ywanggp
"""

import numpy as np
import matplotlib.pyplot as plt

def heuristicSquareAlg(Map, loc, destination, time):
    #get axis range matrix
    MaxAxis = Map.shape
    maxX = MaxAxis[0]
    maxY = MaxAxis[1]
    speed = 0.5
    maxLen = speed * time
    maxLen = int(maxLen)
    xArray = np.linspace(-maxLen, maxLen, 2 * maxLen + 1)
    xArray = np.concatenate((xArray, np.linspace(maxLen - 1, -maxLen + 1, 2 * maxLen - 1)), axis = 0)
    yArray = np.linspace(0, -maxLen, maxLen + 1)
    yArray = np.concatenate((yArray, np.linspace(-maxLen + 1, maxLen, 2 * maxLen)), axis = 0)
    yArray = np.concatenate((yArray, np.linspace(maxLen - 1, 1, maxLen - 1)), axis = 0)
    axisRange = np.column_stack((xArray, yArray))
    axisRange[:, 0] += loc[0]
    axisRange[:, 1] += loc[1]
    axisRange = axisRange.astype(int)
    delete = [True]*len(axisRange)
    for i in range(len(axisRange)):
        if Map[axisRange[i,0],axisRange[i,1]] >= 15:
            delete[i] = False
#    axisRange_tmp = []
#    for i in range(len(axisRange)):
#        if delete[i]:
#            axisRange_tmp += [axisRange[i]]
#    axisRange = np.asarray(axisRange_tmp[:])
    gottenAtTarget = False
    
    #test whether we would touch the margin
    if(all(axisRange[:, 0] >= 0) and all(axisRange[:,0] <= maxX)):
        pass
    else:
        print("x range not in matrix")
        
    if(all(axisRange[:, 1] >= 0) and all(axisRange[:,1] <= maxY)):
        pass
    else:
        print("y range not in matrix")
        
    #test whether the destination is in the square
    if(np.sum(np.abs(destination - loc)) < maxLen):
        finalLoc = destination
        gottenAtTarget = True
        return(finalLoc, gottenAtTarget)
        
    #get the optimal direction
    #test whether x or y are the same
    T = []
    for i in range(axisRange.shape[0]):
        T += [(i,abs(axisRange[i,0] - destination[0])+abs(axisRange[i,1] - destination[1]),
               Map[axisRange[i,0],axisRange[i,1]])]
    T.sort(key=lambda x: x[2])
    T.sort(key=lambda x: x[1])
    finalLoc = axisRange[T[0][0],:]
    return (finalLoc,gottenAtTarget)
    
#    lenX = destination[0] - loc[0]
#    lenY = destination[1] - loc[1]
#    if(abs(lenX) <= maxLen / 2):
#        if(lenY > 0):
#           finalLoc = loc
#           finalLoc[1] += maxLen
#        else:
#           finalLoc = loc
#           finalLoc[1] -= maxLen
#        return(finalLoc, gottenAtTarget)
#
#    
#    if(abs(lenY) <= maxLen / 2):
#        if(lenX > 0):
#           finalLoc = loc
#           finalLoc[0] += maxLen
#        else:
#           finalLoc = loc
#           finalLoc[0] -= maxLen    
#        return(finalLoc, gottenAtTarget)
#
#    #find the best direction
#    dir = 0
#    if(lenX > 0):
#        if(lenY > 0):
#            dir = 3
#        else:
#            dir = 2
#    else:
#        if(lenY > 0):
#            dir = 4
#        else:
#            dir = 1
#    
#    range1 = dir * maxLen - maxLen
#    range2 = dir * maxLen 
#    rangeAxis = axisRange[range1 : range2, :]
#    finalLoc = rangeAxis[int(maxLen / 2), :]
#    finalLoc = finalLoc.astype(int)
#    return(finalLoc, gottenAtTarget)