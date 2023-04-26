#%%
import numpy as np
from scipy.stats import multivariate_normal
import datetime as dt
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def initialize_parameters(X, K):
    # Initialize the means, covariances, and weights
    means = np.random.rand(K, X.shape[1])
    covariances = np.array([np.identity(X.shape[1]) for _ in range(K)])
    weights = np.ones(K) / K
    return means, covariances, weights

def e_step(X, means, covariances, weights, K):
    responsibilities = np.zeros((X.shape[0], K))
    for i in range(K):
        for j in range(X.shape[0]):
            observed = ~np.isnan(X[j])
            responsibilities[j, i] = weights[i] * multivariate_normal.pdf(X[j, observed], mean=means[i, observed], cov=covariances[i][observed][:, observed])
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

def m_step(X, responsibilities, K):
    # Update the means, covariances, and weights
    N_k = np.sum(responsibilities, axis=0)
    means = np.dot(responsibilities.T, X) / N_k[:, None]
    covariances = np.zeros((K, X.shape[1], X.shape[1]))
    for i in range(K):
        X_centered = X - means[i]
        covariances[i] = np.dot(responsibilities[:, i] * X_centered.T, X_centered) / N_k[i]
    weights = N_k / X.shape[0]
    return means, covariances, weights

def em_algorithm(X, K, max_iterations=100, tolerance=1e-4):
    means, covariances, weights = initialize_parameters(X, K)

    for iteration in range(max_iterations):
        means_old, covariances_old, weights_old = means.copy(), covariances.copy(), weights.copy()

        # E-step: Compute the responsibilities using multivariate Gaussian distributions with missing values
        responsibilities = e_step(X, means, covariances, weights, K)

        # M-step: Update the means, covariances, and weights
        means, covariances, weights = m_step(X, responsibilities, K)

        if np.linalg.norm(means - means_old) < tolerance and \
           np.linalg.norm(covariances - covariances_old) < tolerance and \
           np.linalg.norm(weights - weights_old) < tolerance:
            break

    return means, covariances, weights, responsibilities

def impute_missing_values(X, means, covariances, responsibilities):
    # Compute the most probable component for each data point
    most_probable_component = np.argmax(responsibilities, axis=1)

    # Impute the missing values using the conditional Gaussian distribution
    X_imputed = X.copy()
    for i in range(X.shape[0]):
        component = most_probable_component[i]
        missing_values_indices = np.isnan(X[i])
        observed_values_indices = ~missing_values_indices
        
        if np.any(missing_values_indices):  # Check if there are missing values
            mean_missing = means[component, missing_values_indices]
            mean_observed = means[component, observed_values_indices]
            cov_missing = covariances[component][missing_values_indices][:, missing_values_indices]
            cov_observed = covariances[component][observed_values_indices][:, observed_values_indices]
            cov_observed_missing = covariances[component][observed_values_indices][:, missing_values_indices]

            # Compute the conditional mean and covariance for the missing values
            conditional_mean = mean_missing + np.dot(np.dot(cov_observed_missing, np.linalg.inv(cov_observed)),
                                                     (X[i, observed_values_indices] - mean_observed))
            conditional_covariance = cov_missing - np.dot(np.dot(cov_observed_missing, np.linalg.inv(cov_observed)),
                                                           cov_observed_missing.T)

            # Generate missing values from the conditional Gaussian distribution
            X_imputed[i, missing_values_indices] = np.random.multivariate_normal(conditional_mean, conditional_covariance)

    return X_imputed
#%%
K = 3
name = 'AAPL'
funda_data = pd.read_csv('Fundamental_data.csv')
flow_data = funda_data.loc[:,['public_date','pcf','PEG_trailing','TICKER']]
funda_data = funda_data.drop(columns=['qdate',"pcf",'staff_sale','PEG_trailing','gvkey','permno','adate'])
funda_data.set_index('public_date',inplace=True)
funda_data = funda_data.drop_duplicates()
funda_data.reset_index(inplace=True)
print(funda_data)
#%%
funda_stock = funda_data.loc[funda_data.TICKER==name,:]
funda_stock['public_date'] = funda_data.public_date.apply(lambda x:dt.datetime.strptime(x,'%m/%d/%Y'))
flow_stock = flow_data.loc[flow_data.TICKER==name,:]
flow_stock['public_date'] = flow_stock.public_date.apply(lambda x:dt.datetime.strptime(x,'%m/%d/%Y'))
freq = 'Monthly'

price = pd.read_csv('../stock_price/'+freq+'/'+name+'.csv')
price['Date'] = price.Date.apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))
close = price.loc[:,['Date']]
funda_stock = pd.merge(flow_stock,funda_stock,how='outer',on = ['public_date'])
print(funda_stock)
df = pd.merge(funda_stock,close,how='right',left_on=['public_date'],right_on=['Date'])
# Here we merge the dataset together using the forward filling method, which is using the old time data to fill the later time.
df.set_index('Date',inplace=True)
df.drop(columns=['TICKER_x','TICKER_y','public_date'],inplace=True)
#%%
# Run the EM algorithm with missing values
means, covariances, weights, responsibilities = em_algorithm(X, K)

# Impute missing values
X_imputed = impute_missing_values(X, means, covariances, responsibilities)
