#%%
import pandas as pd
import numpy as np

def merge_predictors(stock,freq):
    path = 'predictors/'
    Funda = path+'FundaIndicators/'+freq+'/'+stock+'.csv'
    Macro = path+'Macro/'+freq+'/'+stock+'.csv'
    Tech = path+'Tech/'+freq+'/'+stock+'.csv'

    Funda_stock = pd.read_csv(Funda)
    Macro_stock = pd.read_csv(Macro)
    Tech_stock = pd.read_csv(Tech)
    Funda_stock.set_index('Date',inplace=True)
    Macro_stock.set_index('Date',inplace=True)
    Tech_stock.set_index('Date',inplace=True)
    df = Funda_stock.join(Macro_stock,how='outer')
    df = df.join(Tech_stock)

    df.to_csv(path+'/Merged/'+freq+'/'+stock+'.csv')
# %%
stock_list = ['AAPL','AMZN','BRK-B','GOOG','JNJ','META','MSFT','NVDA','TSLA','V']
for name in stock_list: 
    merge_predictors(name,'Monthly')
    merge_predictors(name,'Weekly')
    merge_predictors(name,'Daily')
# %%
