# Regression Instruction Notebook

## For the regression part, we construct the following structure to show our work:

1. Using the original model parameters to show the parameter tuning result based on the daily datasets.
2. Combined the feature selection with the tuning parameters together to select the best number of features to be selected.
3. Then, based on the feature selection, we want to know which features are selected for each models.
4. Finally, we will apply similar logics into the weekly and monthly frequencies to investigate their model fitting results.
5. Conclusion part: for each model, we make an conclusion for it to let readers understand what we do and our expectation.

## Model Fitting Details
1. Dependent variables:  ${adjusted\,close\,price} _{t+1}$ assuming current time is $t$.
2. Predictors: all the dataset at the time $t$.
3. Time Series Cross Validation:
    - Why we do not use the traditional cross validation method? 
  
    Because they will shuffle the time to make the prediction, which may lead to one situation:
    
    - we will train data from future $(t+1,t+2,t+3)$ to predict current dataset $(t)$
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

    - Technique Details:
    We divide the whole dataset into train dataset and test dataset using the function:

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    
        For the X_train and y_train, we apply the time series cross validation method to select the best feature subsets. 
        
        Then, we evaluate the model using the X_test and y_test to try our best to reduce the overfiting problem.


5. Models:
    We trained Lasso, Ridge Regression and SVR models. For each models, we combined with different feature selection methods. Combinations are as followings:
    - Lasso with Forward Selection 'Lasso.ipynb';
    - Ridge with Forward Selection 'Ridge.ipynb';
    - Lasso with PCA 'Lasso_KnnPca.ipynb';
