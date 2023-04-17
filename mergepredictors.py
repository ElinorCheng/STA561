#%%
import pandas as pd
import numpy as np
#%%
path = 'predictors/'
freq = 'Weekly'
Funda = path+'FundaIndicators/'+freq+'/AAPL.csv'
Macro = path+'Macro/'+freq+'/AAPL.csv'
Tech = path+'Tech/'+freq+'/AAPL.csv'

Funda_stock = pd.read_csv(Funda)
Macro_stock = pd.read_csv(Macro)
Tech_stock = pd.read_csv(Tech)
#%%
Funda_stock.set_index('Date',inplace=True)
Macro_stock.set_index('Date',inplace=True)
Tech_stock.set_index('Date',inplace=True)
df = Funda_stock.join(Macro_stock,how='outer')
df = df.join(Tech_stock)

df.to_csv(path+'/Merged/'+freq+'/AAPL.csv')
# %%
