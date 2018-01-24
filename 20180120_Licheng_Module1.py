import sys
sys.path.append("Functions\Licheng")
from weather_predict import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import *
import xgboost as xgb
import lightgbm as lgb
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

#def my_custom_loss_func1(ground_truth, predictions):
#    ground_truth = map(lambda x: 1 if x>=15 else 0, ground_truth)
#    predictions = map(lambda x: 1 if x>=15 else 0, predictions)
#    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
#    return reduce(lambda x,y: x+y, map(lambda x: 1 if x==0 else 0, diff))*1.0/len(ground_truth)
def my_custom_loss_func2(ground_truth, predictions):
    ground_truth = map(lambda x: 1 if x>=0.5 else 0, ground_truth)
    predictions = map(lambda x: 1 if x>=0.5 else 0, predictions)
    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
    return reduce(lambda x,y: x+y, map(lambda x: 1 if x==0 else 0, diff))*1.0/len(ground_truth)

#score1 = make_scorer(my_custom_loss_func1, greater_is_better=True) 
score2 = make_scorer(my_custom_loss_func2, greater_is_better=True) 
#%%
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_train_copy[:,4:] = 1/(1+np.exp(-(X_train_copy[:,4:]-15)))
X_train_copy = np.concatenate((X_train_copy,X_train_copy[:,4:14].mean(axis = 1).reshape(-1,1),\
                               X_train_copy[:,4:14].std(axis = 1).reshape(-1,1)),axis = 1)
#X_train_copy[:,3] += (X_train_copy[:,2]-1)*24
X_train_copy = np.delete(X_train_copy, [2], axis=1)  
y_train_copy = 1/(1+np.exp(-(y_train_copy-15)))

#xgbr = xgb.XGBRegressor()
#param_grid = {'max_depth':[3,4,5,6],
#              'learning_rate': [0.003,0.03,0.3], 
#              'reg_alpha': [ 0.001, 0.01, 0.1, 1],
#              'reg_lambda': [ 0.01, 0.1, 1, 5]}
#grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score,cv=10,verbose = 2).fit(X_train,y_train)
#xgbr_best = grid_search_xgbr.best_estimator_
#print 'Score for XGBoosting :',cross_val_score(xgbr_best,X_train,y_train,cv=10,scoring=score).mean()

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

#%%
xgbr = xgb.XGBRegressor()
param_grid = {'max_depth':[2],
              'min_child_weight':[1],
              'gamma':[0],
              'subsample':[0.9],
              'colsample_bytree':[0.9],
              'scale_pos_weight':[1],
              'learning_rate': [0.1], 
              'reg_alpha': [1e-5],
              'reg_lambda':[1],
              'objective': ['reg:logistic'],
              'n_estimators':[20]}
##param_grid = {'max_depth':[3],
##              'learning_rate': 0.01*np.linspace(1,10,10), 
##              'reg_alpha': 0.002*np.linspace(1,10,10),
##              'reg_lambda':range(1,11),
##              'objective': ['reg:logistic'],
##              'n_estimators':[100]}
grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score2,cv=3,verbose = 2,iid = False).fit(X_train_copy,y_train_copy)
grid_search_xgbr.grid_scores_
#xgbr_best = grid_search_xgbr.best_estimator_
#print 'Score for XGBoosting :',cross_val_score(xgbr_best,X_train_copy,y_train_copy,cv=3,scoring=score2).mean()


#adb = AdaBoostRegressor()
#print 'Score for AdaBoosting :',cross_val_score(adb,X_train,y_train,cv=3,scoring=score1).mean()
#adb = AdaBoostRegressor()
#param_grid = {'learning_rate': [0.03], 
#              'n_estimators': [100],
#              'loss': ['linear']}
#grid_search_adb = GridSearchCV(adb, param_grid=param_grid, scoring=score2,cv=3,verbose = 2).fit(X_train_copy,y_train_copy)
#adb_best = grid_search_adb.best_estimator_
#print 'Score for AdaBoosting :',cross_val_score(adb_best,X_train_copy,y_train_copy,cv=3,scoring=score2).mean()

#%%
#mlpr = MLPRegressor(hidden_layer_sizes = 100, activation = 'logistic',max_iter = 800)
#print 'Score for  MLPRegressor :',cross_val_score(mlpr,X_train_copy,y_train_copy.ravel(),cv=3,scoring=score2).mean()
#gbr = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3,min_samples_leaf = 12, min_samples_split = 40,\
#                                max_features=8,subsample = 0.8,random_state=10,n_estimators = 80)
#param_grid = {} 
#grid_search_gbr = GridSearchCV(gbr, param_grid=param_grid, scoring=score2,cv=3,verbose = 2,iid = False).fit(X_train_copy,y_train_copy)
#print grid_search_gbr.grid_scores_
#print 'Score for GradientBoostingRegressor:',cross_val_score(gbr,X_train_copy,y_train_copy.ravel(),cv=3,scoring=score2).mean()