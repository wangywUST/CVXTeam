import sys
sys.path.append("Functions\Licheng")
from weather_predict import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV

trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"

#outputPath = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_"+str(3)+".csv" 
#predict = model3(trainPredFile, trainTrueFile, testPredFile)
#predict.to_csv(outputPath,index = False)

#X_train,y_train = model4(trainPredFile, trainTrueFile, testPredFile)

def my_custom_loss_func1(ground_truth, predictions):
    ground_truth = map(lambda x: 1 if x>=15 else 0, ground_truth)
    predictions = map(lambda x: 1 if x>=15 else 0, predictions)
    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
    return reduce(lambda x,y: x+y, map(lambda x: 1 if x==0 else 0, diff))*1.0/len(ground_truth)
def my_custom_loss_func2(ground_truth, predictions):
    ground_truth = map(lambda x: 1 if x>=0.5 else 0, ground_truth)
    predictions = map(lambda x: 1 if x>=0.5 else 0, predictions)
    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
    return reduce(lambda x,y: x+y, map(lambda x: 1 if x==0 else 0, diff))*1.0/len(ground_truth)

score1 = make_scorer(my_custom_loss_func1, greater_is_better=True) 
score2 = make_scorer(my_custom_loss_func2, greater_is_better=True) 

X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_train_copy[:,4:] = 1/(1+np.exp(-(X_train_copy[:,4:]-15)))
y_train_copy = 1/(1+np.exp(-(y_train_copy-15)))

#xgbr = xgb.XGBRegressor()
#param_grid = {'max_depth':[3,4,5,6],
#              'learning_rate': [0.003,0.03,0.3], 
#              'reg_alpha': [ 0.001, 0.01, 0.1, 1],
#              'reg_lambda': [ 0.01, 0.1, 1, 5]}
#grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score,cv=10,verbose = 2).fit(X_train,y_train)
#xgbr_best = grid_search_xgbr.best_estimator_
#print 'Score for XGBoosting :',cross_val_score(xgbr_best,X_train,y_train,cv=10,scoring=score).mean()
##print 'Score for XGBoosting :',cross_val_score(xgbr,X_train[:,range(2)+range(3,14)],y_train,cv=10,scoring=score).mean()

#adb = AdaBoostRegressor()
##param_grid = {'learning_rate': [0.3,1,3,10], 
##              'n_estimators': [50,100,200],
##              'loss': ['linear', 'square', 'exponential']}
#param_grid = {'learning_rate': list(np.linspace(0.5,3,6)), 
#              'n_estimators': [50,100,200],
#              'loss': ['linear', 'square']}
#grid_search_adb = GridSearchCV(adb, param_grid=param_grid, scoring=score,cv=10,verbose = 2).fit(X_train,y_train)
#adb_best = grid_search_adb.best_estimator_
#print 'Score for AdaBoosting :',cross_val_score(adb_best,X_train,y_train,cv=10,scoring=score).mean()


xgbr = xgb.XGBRegressor()
print 'Score for XGBoosting :',cross_val_score(xgbr,X_train,y_train,cv=3,scoring=score1).mean()
xgbr = xgb.XGBRegressor()
#param_grid = {'max_depth':[3,4,5,6],
#              'learning_rate': [0.003,0.03,0.3], 
#              'reg_alpha': [ 0.001, 0.01, 0.1, 1],
#              'reg_lambda': [ 0.01, 0.1, 1, 5]}
param_grid = {'max_depth':[3,4,5,6,7],
              'learning_rate': [0.0003,0.003,0.03], 
              'reg_alpha': [0.1,1,5,10],
              'reg_lambda': range(1,10),
              'objective': ['reg:logistic','reg:linear']}
grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score2,cv=3,verbose = 2).fit(X_train_copy,y_train_copy)
xgbr_best = grid_search_xgbr.best_estimator_
print 'Score for XGBoosting :',cross_val_score(xgbr_best,X_train_copy,y_train_copy,cv=3,scoring=score2).mean()
#adb = AdaBoostRegressor()
#print 'Score for AdaBoosting :',cross_val_score(adb,X_train_copy,y_train_copy,cv=3,scoring=score2).mean()