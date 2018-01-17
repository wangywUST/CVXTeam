import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np


trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
cityLocFile = "C:\Users\lzhaoai\Desktop\predict_weather\CityData.csv"

cityLoc = pd.read_csv(cityLocFile)
xCity = cityLoc['xid']
yCity = cityLoc['yid']
file = trainTrueFile

xsize = 548
ysize = 421

chunksize = xsize * ysize
df = pd.read_csv(file, chunksize = chunksize)
pd.set_option('display.max_rows', None)
#print(df.get_chunk(chunksize))
for i in range(1):
    df_block = df.get_chunk(chunksize)
    windGra = df_block["wind"].map(lambda i: 1 if i >= 15 else 0).values.reshape(xsize,ysize)
    np.save("C:/Users/lzhaoai/Desktop/contest/eWANGPython/Output"+str(i // 18 + 1) + ' Day ' + str(i % 18 + 3) + ' Hour', windGra)
#    x = np.linspace(1, xsize, xsize)
#    y = np.linspace(1, ysize, ysize)
#    X,Y = np.meshgrid(y, x)
#    CS = plt.contour(X, Y, windGra,levels = [15],
#                     colors=('k',),linestyles=('-',),linewidths=(1,))
#    CSF = plt.contourf(X, Y, windGra, 8, alpha=.95, cmap=plt.cm.cool)
#    plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels
#    plt.colorbar(CSF, shrink=0.8, extend='both')
#    plt.scatter(yCity[1:11], xCity[1:11], marker='x', s=50, c = 'gold', zorder=10)
#    plt.scatter(yCity[0], xCity[0], marker='*', s=50, c = 'gold', zorder=10)
    #C = plt.contour(X, Y, windGra, 8, colors='black', linewidth=.1)
#    plt.title(str(i // 18 + 1) + ' Day ' + str(i % 18 + 1) + ' Hour.png')
#    plt.savefig('Output/LichengOutput' + str(i // 18 + 1) + ' Day ' + str(i % 18 + 3) + ' Hour.png')
#    print(i)
#    plt.clf()