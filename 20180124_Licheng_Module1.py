import sys
sys.path.append("Functions\Licheng")
from weather_predict import *
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"
outputPath = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_"+str(5)+".csv"
predict = model5(trainPredFile, trainTrueFile, testPredFile)
predict.to_csv(outputPath,index = False)



