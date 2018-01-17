# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:49:24 2018

@author: lzhaoai
"""
import numpy as np
import matplotlib.pyplot as plt

def showPoint(windGra, xCity, yCity, pathList):
    matShape = windGra.shape
    xsize = matShape[0]
    ysize = matShape[1]
    x = np.linspace(1, xsize, xsize)
    y = np.linspace(1, ysize, ysize)
    X,Y = np.meshgrid(y, x)
    CS = plt.contour(X, Y, windGra,levels = [15],
                     colors=('k',),linestyles=('-',),linewidths=(1,))
    CSF = plt.contourf(X, Y, windGra, 8, alpha=.95, cmap=plt.cm.cool)
    plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels
    plt.colorbar(CSF, shrink=0.8, extend='both')
    plt.scatter(yCity[1:11], xCity[1:11], marker='x', s=50, c = 'gold', zorder=10)
    plt.scatter(pathList[:,1], pathList[:, 0], marker='x', s=5, c = 'gold', zorder=10)
    plt.scatter(yCity[0], xCity[0], marker='*', s=50, c = 'gold', zorder=10)
    plt.show()
    plt.clf()