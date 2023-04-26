import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV,LassoLarsCV,Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,mean_squared_error
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore",category=ConvergenceWarning)
import os
from matplotlib import pyplot as plt
from datetime import datetime

stock_list = ['AAPL','MSFT','GOOG','AMZN','NVDA','BRK-B','TSLA','META','JNJ','V']

Freqs = ["Daily","Weekly", "Monthly"]



def get_data(freq, stock):
    price = pd.read_csv('encode_price/'+freq+'/'+stock+'.csv')
    price = price.sort_values(by='Date').reset_index(drop=True)
    price = price.loc[price.Date>='2010-01-01']
    predictors = pd.read_csv('predictors/Merged/'+freq+'/'+stock+'.csv',index_col='Date')
    NLP = pd.read_csv('predictors/NLP/Daily/NYT_macro_SA.csv').set_index(['Date'])
    predictors = predictors.merge(NLP,left_index=True,right_index=True,how='left')
    predictors = predictors.loc[predictors.index<='2019-12-31',:]
    price.loc[:,'Date'] =[datetime.strptime(str,"%Y-%m-%d") for str in price.loc[:,'Date']]
    predictors.fillna(0,inplace=True)

    return predictors, price


def train_n_split():
    scores = []
    for i in range(3,15):
        tscv = TimeSeriesSplit(n_splits=i)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=False)
        pipe = make_pipeline(MinMaxScaler(),Lasso())

        sfs = SequentialFeatureSelector(pipe,n_jobs = -1,n_features_to_select=6,scoring='neg_root_mean_squared_error')
        sfs.fit(X_train,y_train)
        X_train = sfs.transform(X_train)
        pipmodel = make_pipeline(MinMaxScaler(),LassoCV(cv=tscv))
        pipmodel.fit(X_train,y_train)
        X_test =  sfs.transform(X_test)
        y_pred = pipmodel.predict(X_train)
        score = mean_squared_error(y_pred, y_train)
        print(score)
        scores.append(score)
    n_split = range(3,15)[scores.index(min(scores))]
    return n_split

def train_once(X,y, n_split):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=False)
    tscv = TimeSeriesSplit(n_splits=n_split)
    pipe = make_pipeline(MinMaxScaler(),LassoCV(cv=tscv))
    sfs = SequentialFeatureSelector(pipe,n_jobs = -1,n_features_to_select=6,scoring='neg_root_mean_squared_error')
    sfs.fit(X_train,y_train)
    X_train = sfs.transform(X_train)
    pipmodel = make_pipeline(MinMaxScaler(),LassoCV(cv=tscv))
    pipmodel.fit(X_train,y_train)
    features = sfs.get_feature_names_out()
    X_test =  sfs.transform(X_test)
    y_pred = pipmodel.predict(X_test)
    return y_pred, mean_squared_error(y_pred, y_test),mean_squared_error(y_train, pipmodel.predict(X_train)),features





if __name__ == "__main__":

    warnings.filterwarnings("ignore",category=ConvergenceWarning)
    if not os.path.isdir('Lasso'):
        os.mkdir('Lasso')
    if not os.path.isdir('Lasso/plots'):
        os.mkdir('Lasso/plots')
    if not os.path.isdir('Lasso/results'):
        os.mkdir('Lasso/results')

    print("start")
    features_dic ={}
    mse_df = pd.DataFrame(columns=stock_list)
    for freq in Freqs:
        features_dic[freq] = {}
        mse_results = {}
        for stock in stock_list:
            print(freq+"-"+stock+" : ")
            predictors, price = get_data(freq, stock)
            #predictors = predictors.drop(['gdp','adjusted_close'],axis=1)
            X = predictors.values[:-1,]
            y = price.adjusted_close.shift(-1).values[:len(X)]
            dates = price.Date.shift(-1).values[:len(X)]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=False)
            scores = []
            for i in range(3,15,3):
                y_pred, score_tst,score_tr,features = train_once(X,y, i)
                print(score_tr)
                scores.append(score_tr)
            n_split = scores.index(min(scores))+2
            y_pred, results_tst,results_tr,features = train_once(X,y, n_split=n_split)
            dates = dates[-len(y_test):]
            feature_names = []
            mse_results[stock] = results_tst
            
            for f in features:
                feature_names.append(predictors.columns[int(f[1:])])
            features_dic[freq][stock] = feature_names

        
            if not os.path.isdir("Lasso/plots/"+freq):
                os.mkdir("Lasso/plots/"+freq)
            if not os.path.isdir("Lasso/results/"+freq):
                os.mkdir("Lasso/results/"+freq)
            plt.figure()
            plt.plot(dates, y_pred,label="Predictions")
            plt.plot(dates, y_test,label="Actual Data")
            plt.title(stock+" - Lasso")
            plt.legend()
            plt.savefig("Lasso/plots/"+freq+"/"+stock+".png")

        mse_df = mse_df.append(mse_results,ignore_index=True)

        features_df = pd.DataFrame(features_dic[freq])
        print(features_df)
        
        features_df.to_csv("Lasso/results/"+freq+"/"+stock+".csv",index=0)
    print(mse_df)
    mse_df.to_csv("Lasso/MSE.csv",index=0)
            
        
    





            

            

