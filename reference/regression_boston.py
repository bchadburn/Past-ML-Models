
# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
import numpy as np
import matplotlib.pyplot as plt

# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())
summary = boston_input.describe()
summary.head()

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# the histogram of the data to get a sense of the respone variable
n, bins, patches = plt.hist(prelim_model_data[:,0], 10, normed=1, facecolor='green', alpha=0.75)
# We see a right skew. So we will ahead and use the log trasnform of the 
# response variable.

# Perform Log transformation
from numpy import log
prelim_model_data[:,0] = log(prelim_model_data[:,0])

# Splitting the dataset into the Training set and Test set
X = prelim_model_data[:,1:13]
y = prelim_model_data[:, 0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

# Scale Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler2 = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler2.fit_transform(y_train.reshape(-1, 1))
X_test = scaler.transform(X_test)
y_test = scaler2.transform(y_test.reshape(-1, 1))
model_data = np.concatenate((y_train, X_train), axis=1)
 
#Investigatin Correlations and multicollinearity issues   

# Correlation Matrix
import seaborn as sns
corr = pd.DataFrame(model_data).corr(method='pearson')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# Get top absolute correlations
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = pd.DataFrame(model_data).columns
    for i in range(0, model_data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = pd.DataFrame(model_data).corr(method='pearson').abs().unstack()
    labels_to_drop = get_redundant_pairs(model_data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(model_data, 10))


# Define Ridge Regression
 
ridge_reg = Ridge(alpha=5, solver="cholesky", random_state=42)
ridge_reg.fit(X_train, y_train)


# We aren't seeing much improvement. However, sense we potentially have unuseful 
# variables we should try Lasso or Elastic net. We will use elastic and 
elastic_net = ElasticNet(alpha=.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'alpha': [.1, .5, 1, 3, 5, 8, 10,]},
  ]
grid_searchRR = GridSearchCV(ridge_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_searchRR.fit(X_train, y_train)

grid_searchRR.best_params_
grid_searchRR.best_estimator_


param_grid = [
    {'alpha': [.1, .5, 1], 'l1_ratio': [.2, .3, .5, .8]},
  ]
grid_searchEN = GridSearchCV(elastic_net, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_searchEN.fit(X_train, y_train)

grid_searchEN.best_params_
grid_searchEN.best_estimator_


SET_FIT_INTERCEPT = True

regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT), 
              Ridge(alpha = 5, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED),
               Lasso(alpha = 0.1, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),
               ElasticNet(alpha = 0.1, l1_ratio = 0.2, 
                          max_iter=10000, tol=0.01, 
                          fit_intercept = SET_FIT_INTERCEPT, 
                          normalize = False, 
                          random_state = RANDOM_SEED)]

# --------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
# As an alternative to 10-fold cross-validation, restdata with its 
# small sample size could be analyzed would be a good candidate
# for  leave-one-out cross-validation, which would set the number
# of folds to the number of observations in the data set.
N_FOLDS = 10

# set up numpy array for storing results
names = ["LinearRegression", "Ridge", "Lasso","ElasticNet"]
cv_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   the structure of modeling data for this study has the
#   response variable coming first and explanatory variables later          
#   so 1:model_data.shape[1] slices for explanatory variables
#   and 0 is the index for the response variable    
#    X_train = model_data[train_index, 1:model_data.shape[1]]
#    X_test = model_data[test_index, 1:model_data.shape[1]]
#    y_train = model_data[train_index, 0]
#    y_test = model_data[test_index, 0]   
# =============================================================================
#     print('\nShape of input data for this fold:',
#           '\nData Set: (Observations, Variables)')
#     print('X_train:', X_train.shape)
#     print('X_test:',X_test.shape)
#     print('y_train:', y_train.shape)
#     print('y_test:',y_test.shape)
# 
# =============================================================================
    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold
        print('Fitted regression intercept:', reg_model.intercept_)
        print('Fitted regression coefficients:', reg_model.coef_)
 
        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
print(cv_results_df.mean())  


