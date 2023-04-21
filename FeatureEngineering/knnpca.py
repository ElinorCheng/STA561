import numpy as np
import pandas as pd
import sys
from scipy.sparse import dia_matrix
import scipy.sparse.linalg
import math
from pca import pca

from sklearn.decomposition import PCA
from fancyimpute import KNN

df = pd.read_csv("../predictors/Merged/Weekly/AAPL.csv", names=["Date","dpr","npm","gpm","roa","roe","capital_ratio","de_ratio","cash_ratio","curr_ratio","inv_turn","pay_turn","sale_nwc","rd_sale","accrual","adjusted_close","gdp","gdpr1","gdpr2","cpi","bond20yr","bond30yr","fedfunds","cpir","wpir","unemp","employ","SMA","EMA","STOCH_k","STOCH_d","RSI","MFI","SAR","AD"])
df = df[df.Date<="2017-06-30"]

from sklearn.preprocessing import StandardScaler

features = ["dpr","npm","gpm","roa","roe","capital_ratio","de_ratio","cash_ratio","curr_ratio","inv_turn","pay_turn","sale_nwc","rd_sale","accrual","adjusted_close","gdp","gdpr1","gdpr2","cpi","bond20yr","bond30yr","fedfunds","cpir","wpir","unemp","employ","SMA","EMA","STOCH_k","STOCH_d","RSI","MFI","SAR","AD"]

x = df.loc[:, features].values

x = StandardScaler().fit_transform(x)

x = KNN(k=5).fit_transform(x)

n = 8

pca = PCA(n_components=n)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component ' + str(i) for i in range(1,n+1)])