import sys
sys.path.append("Functions\Licheng")
from weather_predict import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import *
from xgboost import *
import lightgbm as lgb
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV

trainPredFile = "C:\Users\wangyw\Desktop\Data\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\wangyw\Desktop\Data\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\wangyw\Desktop\Data\ForecastDataforTesting_201712.csv"

# trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
# trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
# testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"

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
X_train_copy = np.delete(X_train_copy, [2], axis=1)
y_train_copy = 1/(1+np.exp(-(y_train_copy-15)))

xgbr = xgb.XGBRegressor()
param_grid = {'max_depth':range(2,6),
           'learning_rate': 3*np.logspace(-6,-1,2),
           'reg_alpha': np.logspace(-6,-1,2),
           'reg_lambda': range(4,8),
           'objective': ['reg:linear','reg:logistic'],
           'n_estimators':[50,100,200]}

grid_search_xgbr = GridSearchCV(xgbr, param_grid=param_grid, scoring = score2,cv=3,verbose = 2).fit(X_train_copy,y_train_copy)
xgbr_best = grid_search_xgbr.best_estimator_
print 'Score for XGBoosting :', cross_val_score(xgbr_best, X_train_copy, y_train_copy, cv = 3, scoring = score2).mean()