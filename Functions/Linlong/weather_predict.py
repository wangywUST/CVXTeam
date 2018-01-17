import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

def model1(trainPredFile, trainTrueFile, testPredFile, xsize = 548, ysize = 421):
    chunksize = xsize * ysize * 10
    df_train = pd.read_csv(trainPredFile, chunksize = chunksize)
    df_train_true = pd.read_csv(trainTrueFile, chunksize = chunksize / 10)
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    
    
    clf_p = MLPRegressor(random_state = 0)
    predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])
    for i in range(90): 
        X_train = df_train.get_chunk(chunksize).drop(["date_id","hour"], axis = 1)
        X_train = pd.get_dummies(X_train, columns = ["model"]).values
        X_test = df_test.get_chunk(chunksize).drop(["date_id","hour"], axis = 1)
        X_test = pd.get_dummies(X_test, columns = ["model"]).values
        Data= df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
        y_train = Data["wind"].values.reshape(-1,1)
        Data.drop(["wind"],axis = 1,inplace = True)
        y_train = np.kron(y_train,np.ones(10).reshape(-1,1)).ravel()   
        print "block "+str(i)+" loaded"
        clf_p.partial_fit(X_train,y_train)
        y_test = clf_p.predict(X_test)
        y_test = np.mean(y_test.reshape(-1,10),axis = 1)
        wind = pd.DataFrame(y_test,columns = ["wind"])
        predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)
        print "block "+str(i)+" predicted"
    return predict

def model2(trainPredFile, trainTrueFile, testPredFile, xsize = 548, ysize = 421):
    chunksize = xsize * ysize * 10
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    df_train_true = pd.read_csv(trainTrueFile, chunksize = chunksize / 10)
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])
    for i in range(90): 
        X_test = df_test.get_chunk(chunksize)
        y_pred = X_test["wind"].values.reshape(xsize * ysize, 10)
        y_pred1 = np.median(y_pred, axis = 1).ravel()
        y_pred2 = np.mean(y_pred, axis = 1).ravel()
        y_pred = np.max([y_pred1,y_pred2],axis = 0)
        Data = df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
        Data.drop(["wind"],axis = 1,inplace = True)
        wind = pd.DataFrame(y_pred,columns = ["wind"])
        predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)
    return predict

def model3(trainPredFile, trainTrueFile, testPredFile, xsize = 548, ysize = 421):
    chunksize = xsize * ysize * 10
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    df_train_true = pd.read_csv(trainTrueFile, chunksize = chunksize / 10)
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])
    for i in range(90): 
        X_test = df_test.get_chunk(chunksize)
        y_pred = X_test["wind"].values.reshape(xsize * ysize, 10)
        y_pred = np.max(y_pred, axis = 1).ravel()
        Data = df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
        Data.drop(["wind"],axis = 1,inplace = True)
        wind = pd.DataFrame(y_pred,columns = ["wind"])
        predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)
    return predict