# Classification Instruction Notebook

## For the Classification part, we construct the following structure to show our work:
1. We generally fit the dependent variable named 'Direction', which is constructed throught $ \textbf{1} _{(AdjustedClosePrice_{t} - AdjustedClosePrice_{t-1}) > 0} $. The result is not ideal for the classification model.
2. We create a new depdendent variable named ’Direction2‘, which is constructed throught $ \textbf{1} _{(AdjustedClosePrice_{t+13} - AdjustedClosePrice_{t-1}) > 0} $. Use the classification model to fit it. The result improved a lot.
3. So, we want to do the feature selection, which should investigate the best subset to show the power of classification prediction for each stocks.
4. We apply the same investigation on the weekly and monthly frequencies. For each model, the fitting results varys. 


## Model Fitting Details
1. Dependent variables: 
    - Direction: $ \textbf{1} _{(AdjustedClosePrice_{t} - AdjustedClosePrice_{t-1}) > 0} $
    - Direction 2: $ \textbf{1} _{(AdjustedClosePrice_{t+13} - AdjustedClosePrice_{t-1}) > 0} $

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
    The feature selection method is the forward selection. Under the regression problem, we use the weighted F1-score as the scoring method to select the best feature subsets. We let the function to decide which number would be an optimal number for each stocks under each models. After that, the investigate the feature selection statistics.
    - Why weighted F1-score? It takes F1-score under each classes into consideration. If the F1-score for each class is not balanced. The evaluation method will add penalties the final classification result, which will decrease the performance score. 

    - Technique Details:
    We divide the whole dataset into train dataset and test dataset using the function:

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    
        For the X_train and y_train, we apply the time series cross validation method to select the best feature subsets. 
        
        Then, we evaluate the model using the X_test and y_test to try our best to reduce the overfiting problem.



5. Models:
    We trained SVM, Random Forest and Logistic Regression. For each models, we combined with forward feature selection methods. Combinations are as followings:
    - SVM 'SVM.ipynb'
    - Random Forest & XGboost 'Random_Forest.ipynb'
    - Logistic Regression 'Logistic_Regression.ipynb'
