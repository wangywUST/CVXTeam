# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt

def obtainScore(pathList, windGraph):
    timeLen = pathList.shape[0]
    threshold = 15
    seg = len(pathList)/30
    flag = False
    for j in range(seg):   
        for i in range(30):
            if(windGraph[j,pathList[i, 0], pathList[i, 1]] >= threshold):
                print("die " + str(windGraph[j,pathList[i, 0], pathList[i, 1]]) +" " + str(j)+ " " +str(i))
                flag = True
    #            plt.scatter(pathList[i, 1], pathList[i, 0], marker='o', s=100, c = 'gold', zorder=10) 
    return 1440 if flag else (timeLen - 1) * 2