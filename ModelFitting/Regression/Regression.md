# Regression Instruction Notebook

## For the regression part, we construct the following structure to show our work:

1. Using the original model parameters to show the parameter tuning result based on the daily datasets.
2. Combined the feature selection with the tuning parameters together to select the best number of features to be selected.
3. Then, based on the feature selection, we want to know which features are selected for each models.
4. Finally, we will apply similar logics into the weekly and monthly frequencies to investigate their model fitting results.
5. Conclusion part: for each model, we make an conclusion for it to let readers understand what we do and our expectation.

## Model Fitting Details
1. Dependent variables: adjusted close price_{t+1} assuming current time is t.
2. Predictors: all the dataset at the time t.
3. Time Series Cross Validation:
    - Why we do not use the traditional cross validation method? 
    
    Because they will shuffle the time to make the prediction, which may lead to one situation:
    
    - we will train data from future(t+1,t+2,t+3) to predict current dataset (t)
    So, we use the time series cross validation method to train our model:

                from sklearn.model_selection import TimeSeriesSplit

                X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
                y = np.array([1, 2, 3, 4, 5, 6])
                tscv = TimeSeriesSplit(n_splits=3)
                print(tscv)  
                TimeSeriesSplit(n_splits=3)
                for train, test in tscv.split(X):
                    print("%s %s" % (train, test))
                [0 1 2] [3]
                [0 1 2 3] [4]
                [0 1 2 3 4] [5]
    
    From there, you can find that training dataset will always be the time before the test dataset. Using the timeseries cross validation method, we can guarantee that we always use the past information to predict the future.


4. Feature Selection:
    The feature selection method is the forward selection. Under the regression problem, we use the MSE as the scoring method to select the best feature subsets. We go through different feature amounts to select the best feature amounts to make the regression prediction. The optimal feature numbers will vary depends on the model.

