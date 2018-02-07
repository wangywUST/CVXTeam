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

trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201802.csv"
trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201802.csv"
testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201802.csv"


#X_train,y_train_wind,y_train_rainfall = model5(trainPredFile, trainTrueFile, testPredFile)
#indshow = (X_train[:,0] == 142) & (X_train[:,1] == 328) & (X_train[:,3] == 3)
#%%
#def my_custom_score_func_org(ground_truth, predictions):
#    ground_truth = map(lambda x: 1 if x>=15 else 0, ground_truth)
#    predictions = map(lambda x: 1 if x>=15 else 0, predictions)
#    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
#    return reduce(lambda x,y: x+y, map(lambda x: 1 if x<=0 else 0, diff))*1.0/len(ground_truth)
def my_custom_score_func(ground_truth, predictions):
    ground_truth = map(lambda x: 1 if x>=0.5 else 0, ground_truth)
    predictions = map(lambda x: 1 if x>=0.5 else 0, predictions)
    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
    return reduce(lambda x,y: x+y, map(lambda x: 1 if x==0 else 0, diff))*1.0/len(ground_truth)

#score_org = make_scorer(my_custom_score_func_org, greater_is_better=True) 
score = make_scorer(my_custom_score_func, greater_is_better=True) 
#%%
X_train_copy = X_train.copy()
y_train_wind_copy = y_train_wind.copy()
y_pred_wind_rough = np.max([np.mean(X_train_copy[:,4:14],axis = 1),\
                       np.median(X_train_copy[:,4:14],axis = 1)],axis = 0)
y_pred_wind_rough = 1/(1+np.exp(-(y_pred_wind_rough-15)))
y_train_rainfall_copy = y_train_rainfall.copy()
y_pred_rainfall_rough = np.max([np.mean(X_train_copy[:,14:],axis = 1),\
                       np.median(X_train_copy[:,14:],axis = 1)],axis = 0)
y_pred_rainfall_rough = 1/(1+np.exp(-(y_pred_rainfall_rough-4)))
X_train_copy[:,4:14] = 1/(1+np.exp(-(X_train_copy[:,4:14]-15))) 
X_train_copy[:,14:] = 1/(1+np.exp(-(X_train_copy[:,14:]-4))) 
dist = (np.abs(X_train_copy[:,0] - 142) + np.abs(X_train_copy[:,1] - 328)).reshape(-1,1)
ind1 = (X_train_copy[:,2] == 4)
X_train_copy = np.delete(X_train_copy, [0,1,2], axis=1)
X_train_copy = np.concatenate((X_train_copy[:,0].reshape(-1,1),dist,X_train_copy[:,1:]),axis = 1) 
del dist
y_train_wind_copy = 1/(1+np.exp(-(y_train_wind_copy-15)))
y_train_rainfall_copy = 1/(1+np.exp(-(y_train_rainfall_copy-4)))

#num1,num2 = 20,20
#ind1 = (num1<= X_train_copy[:,0]) & (X_train_copy[:,0] <= num2)
#ind2 = (y_pred_rough >= 0.7) | (y_pred_rough <= 0.3)
#ind2 = (y_pred_rough < 0.7) & (y_pred_rough > 0.3)
#ind2 = ((18<= X_train_copy[:,0]) & (X_train_copy[:,0] <= 20)) | \
#        (X_train_copy[:,0] == 13) | (X_train_copy[:,0] == 14) | (X_train_copy[:,0] == 16)
##ind = ((9<= X_train_copy[:,0]) & (X_train_copy[:,0] <= 12)) | \
##        (X_train_copy[:,0] == 15) | (X_train_copy[:,0] == 17)
ind = ind1 #& ind2#| ind2
print  my_custom_score_func(y_train_wind_copy[ind], y_pred_wind_rough[ind])
print  my_custom_score_func(y_train_rainfall_copy[ind], y_pred_rainfall_rough[ind])
#%%

#xgbr = xgb.XGBRegressor()
#
#print cross_val_score(xgbr,X_train_copy[ind][:,:12],y_train_wind_copy[ind],cv=3,scoring=score,verbose = 2).mean()
#print cross_val_score(xgbr,X_train_copy[ind][:,range(2)+range(12,22)],y_train_rainfall_copy[ind],cv=3,scoring=score,verbose = 2).mean()

#xgbr = xgb.XGBRegressor()
#param_grid = {'max_depth':[3,4,5,6],
#              'learning_rate': [0.003,0.03,0.3], 
#              'reg_alpha': [ 0.001, 0.01, 0.1, 1],
#              'reg_lambda': [ 0.01, 0.1, 1, 5]}
#grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score,cv=10,verbose = 2).fit(X_train,y_train)
#xgbr_best = grid_search_xgbr.best_estimator_
#print 'Score for XGBoosting :',cross_val_score(xgbr_best,X_train,y_train,cv=10,scoring=score).mean()


#%%
#xgbr = xgb.XGBRegressor()
##param_grid = {}
#param_grid = {'max_depth':[7],
#              'min_child_weight':[2],
#              'gamma':[0],
#              'subsample':[0.7],
#              'colsample_bytree':[0.7],
#              'colsample_bylevel':[0.7],
#              'scale_pos_weight':[1],
#              'learning_rate': [0.1], 
#              'reg_alpha': [1e-5],
#              'reg_lambda':[1],
#              'objective': ['reg:logistic'],
#              'n_estimators':[100]}
#grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring=score,cv=3,verbose = 0,iid = False).fit(X_train_copy[ind],y_train_copy[ind])
##print grid_search_xgbr.grid_scores_
#xgbr_best = grid_search_xgbr.best_estimator_
#xgbr_best.fit(X_train_copy[ind],y_train_copy[ind])
##print 'Score for XGBoosting :',cross_val_score(xgbr,X_train_copy,y_train_copy,cv=3,scoring=score,verbose = 2).mean()
#print cross_val_score(xgbr,X_train_copy[ind],y_train_copy[ind],cv=3,scoring=score,verbose = 2).mean()
##print my_custom_score_func(y_train_copy[ind],xgbr_best.predict(X_train_copy[ind]))

##%%
#xgbc = xgb.XGBClassifier()
##param_grid = {}
#param_grid = {'max_depth':[7],
#              'min_child_weight':[2],
#              'gamma':[0],
#              'subsample':[0.8],
#              'colsample_bytree':[0.8],
#              'colsample_bylevel':[0.8],
#              'scale_pos_weight':[1],
#              'learning_rate': [0.1], 
#              'reg_alpha': [1e-5],
#              'reg_lambda':[1],
#              'objective': ['reg:logistic'],
#              'n_estimators':[100]}
#X_t,y_t = X_train_copy[ind],y_train_copy[ind]
#X_t[:,2:] = np.round(X_t[:,2:]).astype(int)
#y_t = np.round(y_t).astype(int).ravel()
##%%
#grid_search_xgbc = GridSearchCV(xgbc, param_grid=param_grid, scoring=score,cv=3,verbose = 0,iid = False).fit(X_t,y_t)
##print grid_search_xgbr.grid_scores_
#xgbc_best = grid_search_xgbc.best_estimator_
#xgbc_best.fit(X_t,y_t)
##print 'Score for XGBoosting :',cross_val_score(xgbr,X_train_copy,y_train_copy,cv=3,scoring=score,verbose = 2).mean()
##print cross_val_score(xgbr,X_train_copy[ind],y_train_copy[ind],cv=3,scoring=score,verbose = 2).mean()
#print my_custom_score_func(y_t,xgbc_best.predict(X_t))