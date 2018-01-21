import sys
sys.path.append("Functions\Licheng")
from weather_predict import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# trainPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTraining_201712.csv"
# trainTrueFile = "C:\Users\lzhaoai\Desktop\predict_weather\In_situMeasurementforTraining_201712.csv"
# testPredFile = "C:\Users\lzhaoai\Desktop\predict_weather\ForecastDataforTesting_201712.csv"

trainPredFile = "C:\Users\wangyw\Desktop\Data\ForecastDataforTraining_201712.csv"
trainTrueFile = "C:\Users\wangyw\Desktop\Data\In_situMeasurementforTraining_201712.csv"
testPredFile = "C:\Users\wangyw\Desktop\Data\ForecastDataforTesting_201712.csv"

#outputPath = "C:\Users\lzhaoai\Desktop\predict_weather\predict_model_"+str(3)+".csv"
#predict = model3(trainPredFile, trainTrueFile, testPredFile)
#predict.to_csv(outputPath,index = False)

#X_train,y_train = model4(trainPredFile, trainTrueFile, testPredFile)

def my_custom_loss_func(ground_truth, predictions):
    ground_truth = map(lambda x: 1 if x>=15 else 0, ground_truth)
    predictions = map(lambda x: 1 if x>=15 else 0, predictions)
    diff = [ground_truth[i] - predictions[i] for i in range(len(ground_truth))]
    return reduce(lambda x,y: x+y, map(lambda x: 1 if x!=0 else 0, diff))*1.0/len(ground_truth)
ground_truth = [2,3,44,6]
predictions = [5,23,5,7]
print my_custom_loss_func(ground_truth, predictions)
#    diff = np.abs(ground_truth - predictions).max()
#    return np.log(1 + diff)

#xgbr = xgb.XGBRegressor()
#print 'Score for XGBoosting :',cross_val_score(xgbr,X_train,y_train,cv=10,scoring='accuracy').mean()



#clf_p = MLPRegressor(random_state = 0)
#predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])
#for i in range(90):
#    X_train = df_train.get_chunk(chunksize).drop(["date_id","hour"], axis = 1)
#    X_train = pd.get_dummies(X_train, columns = ["model"]).values
#    X_test = df_test.get_chunk(chunksize).drop(["date_id","hour"], axis = 1)
#    X_test = pd.get_dummies(X_test, columns = ["model"]).values
#    Data= df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
#    y_train = Data["wind"].values.reshape(-1,1)
#    Data.drop(["wind"],axis = 1,inplace = True)
#    y_train_org = y_train.copy().ravel()
#    y_train = np.kron(y_train,np.ones(10).reshape(-1,1)).ravel()
#    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
#    for train_index, val_index in rs.split(X_train):
#        X_train_train,y_train_train = X_train[train_index],y_train[train_index]
#        X_train_val,y_train_val = X_train[val_index],y_train[val_index]
#        clf_p.partial_fit(X_train_train,y_train_train)
#        insamplemse = ((clf_p.predict(X_train_train) - y_train_train)**2).mean()
#        outsamplemse = ((clf_p.predict(X_train_val) - y_train_val)**2).mean()
#        y_pred = clf_p.predict(X_train)
#        y_pred = np.mean(y_pred.reshape(-1,10),axis = 1)
#        allsamplemse = ((y_pred - y_train_org)**2).mean()
#        print "day "+str(i/18)+", hour "+str(3+i%18)+", total MSE: "+str("{:.4f}".format(insamplemse))\
#        +" "+str("{:.4f}".format(outsamplemse))+" "+str("{:.4f}".format(allsamplemse))
#    y_test = clf_p.predict(X_test)
#    y_test = np.mean(y_test.reshape(-1,10),axis = 1)
#    wind = pd.DataFrame(y_test,columns = ["wind"])
#    predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)