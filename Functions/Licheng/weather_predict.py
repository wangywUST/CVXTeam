import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from jumpDays import *

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

def model4(trainPredFile, trainTrueFile, testPredFile, xsize = 548, ysize = 421):
    chunksize = xsize * ysize * 10
    df_train = pd.read_csv(trainPredFile, chunksize = chunksize)
    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
    df_train_true = pd.read_csv(trainTrueFile, chunksize = chunksize / 10)
#    predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])  
    blksz = 10
    xseg = [blksz]*(xsize/blksz)+[xsize%blksz]
    yseg = [blksz]*(ysize/blksz)+[ysize%blksz]
    for x in range(len(xseg)):
        for y in range(len(yseg)):
            X_blk,y_blk = [],[]
            for i in range(90): 
                wind = df_train.get_chunk(chunksize)
                wind = wind["wind"].values.reshape(xsize * ysize, 10)
                Data = df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
                y_train = Data["wind"].values.reshape(-1,1)
                X_train = Data.drop(["wind"],axis = 1)
                del Data
                X_train = np.concatenate((X_train,wind),axis = 1)
                ind = (sum(xseg[:x]) < X_train[:,0]) & (X_train[:,0] <= sum(xseg[:x+1])) & \
                (sum(yseg[:y]) < X_train[:,1]) & (X_train[:,1] <= sum(yseg[:y+1]))
                X_train_mini,y_train_mini = X_train[ind],y_train[ind]
                print "block "+str(i)+" loaded"
                X_blk += [X_train_mini]
                y_blk += [y_train_mini]
            return np.asarray(X_blk).reshape(-1,14),np.asarray(y_blk).reshape(-1,1)
#        Data = df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
#        Data.drop(["wind"],axis = 1,inplace = True)
#        wind = pd.DataFrame(y_pred,columns = ["wind"])
#        predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)
#    return predict


#def model5(trainPredFile, trainTrueFile, testPredFile, xsize = 548, ysize = 421):
#    chunksize = xsize * ysize * 10
##    df_test = pd.read_csv(testPredFile, chunksize = chunksize)
#    
#    
#    clf_p = MLPRegressor(random_state = 0)
#    predict = pd.DataFrame(columns = ["xid","yid","date_id","hour","wind"])
#    for dayNum in [0]:#range(5):
#        df_d = pd.read_csv(trainPredFile, chunksize = chunksize)
#        df_d = jumpDays(df_d, dayNum-1, chunksize)
#        df_fast_d,df_mid_d,df_slow_d = df_d,df_d,df_d 
#        df_l = pd.read_csv(trainTrueFile, chunksize = chunksize/10)
#        df_l = jumpDays(df_l, dayNum-1, chunksize)
#        df_fast_l,df_mid_l,df_slow_l = df_l,df_l,df_l       
#        for time in [0]:#range(18):
#            TrainData = []
#            TrainLabel = []
#            if time == 0:
#                 TrainData = df_fast_d.get_chunk(chunksize).drop(["date_id"], axis = 1)
#                 TrainData = pd.get_dummies(TrainData, columns = ["model"]).values
#                 X_train = TrainData.copy()             
#                 TrainData = df_fast_d.get_chunk(chunksize).drop(["date_id"], axis = 1)  
#                 TrainData = pd.get_dummies(TrainData, columns = ["model"]).values
#                 X_train = np.concatenate((X_train,TrainData),axis = 0)
#                 
#                 TrainLabel = (df_fast_l.get_chunk(chunksize / 10).reset_index(drop=True))["wind"].values.reshape(-1,1)
#                 TrainLabel = np.kron(TrainLabel,np.ones(10).reshape(-1,1)).ravel()
#                 y_train = TrainLabel.copy()
#                 TrainLabel = (df_fast_l.get_chunk(chunksize / 10).reset_index(drop=True))["wind"].values.reshape(-1,1)
#                 TrainLabel = np.kron(TrainLabel,np.ones(10).reshape(-1,1)).ravel()  
#                 y_train = np.concatenate((y_train,TrainLabel),axis = 0)
#                 print "block "+str(time)+" loaded"
##    for dayNum in range(5):
##        TestData = []
##        Output = []
##        for time in range(18):
##            X_test = df_test.get_chunk(chunksize).drop(["date_id","hour"], axis = 1)
##            X_test = pd.get_dummies(X_test, columns = ["model"]).values
##            TestData += [X_test]
##            Train_true = df_train_true.get_chunk(chunksize / 10).reset_index(drop=True)
##            Train_true.drop(["wind"],axis = 1,inplace = True)
##            Output += [Train_true]
##            print "block "+str(time)+" loaded"
#    return X_train,y_train
##        clf_p.partial_fit(X_train,y_train)
##        y_test = clf_p.predict(X_test)
##        y_test = np.mean(y_test.reshape(-1,10),axis = 1)
##        wind = pd.DataFrame(y_test,columns = ["wind"])
##        predict = predict.append(pd.concat((Data,wind),axis = 1),ignore_index = True)
##        print "block "+str(i)+" predicted"
##    return predict