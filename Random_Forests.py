
# coding: utf-8

# In[1]:


import os
#Set working directory
os.chdir('D:\\MSDS\\422 ML\\Boston Housing')
os.getcwd()


# In[2]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 5

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True



# In[3]:


import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns # plotting, including heat map
import sklearn
from sklearn.preprocessing import power_transform
import scipy
from scipy.stats import uniform  # for training-and-test split
import statsmodels.api as sm  # statistical models (including regression)
import statsmodels.formula.api as smf  # R-like model specification
from sklearn.tree import DecisionTreeRegressor  # machine learning tree
from sklearn.ensemble import RandomForestRegressor # ensemble method
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[4]:


import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')


# In[6]:


# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())


# In[7]:


print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())


# In[8]:


# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())


# In[9]:


print(boston_input.columns)


# Looking at the variable types to ensure they have the right variable type. For example, rooms which shows as float, should be set as an integer. Its not a big deal here as they are all numeric, but it sometimes its best to cast/set them so we can use them for the models. 

# In[10]:


# correlation heat map setup for seaborn
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)


# In[11]:


# examine intercorrelations with correlation matrix/heat map
corr_chart(df_corr = boston) 


# In[12]:


# Creating df with relevant data
boston_input_df = boston_input.loc[:, ('crim', 'zn', 'indus', 'chas', 'nox', 'rooms', 'age',
                                       'dis', 'rad', 'tax', 'ptratio', 'lstat', 'mv' )]


# In[13]:


boston_input_df_labels = [
    'crim', 
    'zn', 
    'indus',
    'chas', 
    'nox',
    'rooms',
    'age',
    'dis', 
    'rad', 
    'tax', 
    'ptratio', 
    'lstat', 
    'mv'       
]


# In[14]:


for i in range(12):
            file_title = boston_input_df.columns[i] + '_and_' + 'mv'
            plot_title = boston_input_df.columns[i] + ' and ' + 'mv'
            fig, axis = plt.subplots()
            axis.set_xlabel(boston_input_df_labels[i])
            axis.set_ylabel('mv')
            plt.title(plot_title)
            scatter_plot = axis.scatter(boston_input_df[boston_input_df.columns[i]], 
            boston_input_df['mv'], 
            facecolors = 'none', 
            edgecolors = 'blue')
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)


# In[15]:


# the histogram of the data to get a sense of the respone variable
n, bins, patches = plt.hist(boston_input_df['mv'], 10, density=1, facecolor='green', alpha=0.75)
plt.title('Histogram of Median Home Value')
plt.xlabel('1970 dollars (in thousands)')
plt.ylabel('Count')
print('skewness:', scipy.stats.skew(boston_input_df['mv'], axis=0, bias=True))
print('kurtosis:', scipy.stats.kurtosis(boston_input_df['mv'], axis=0, fisher=True, bias=True))


# We see a right skew. So we will go ahead and use the log transform of the response variable

# In[16]:


# Perform Log transformation
from numpy import log
logresponse = log(boston_input_df['mv'])


# In[17]:


# the histogram of the data to get a sense of the respone variable
n, bins, patches = plt.hist(logresponse, 10, density=1, facecolor='green', alpha=0.75)
plt.title('Histogram of Log Transformed Median Home Value')
plt.xlabel('1970 dollars (in thousands)')
plt.ylabel('Count')
print('skewness:', scipy.stats.skew(logresponse, axis=0, bias=True))
print('kurtosis:', scipy.stats.kurtosis(logresponse, axis=0, fisher=True, bias=True))


# In[18]:


boston_input_df['logmv'] = log(boston_input_df['mv'])


# In[19]:


for i in range(12):
            file_title = boston_input_df.columns[i] + '_and_' + 'logmv'
            plot_title = boston_input_df.columns[i] + ' and ' + 'logmv'
            fig, axis = plt.subplots()
            axis.set_xlabel(boston_input_df_labels[i])
            axis.set_ylabel('logmv')
            plt.title(plot_title)
            scatter_plot = axis.scatter(boston_input_df[boston_input_df.columns[i]], 
            boston_input_df['logmv'], 
            facecolors = 'none', 
            edgecolors = 'blue')
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)


# This does seem to have linearized lstat, crim and indus in respect to the response - mv. Age, dis and maybe lstat still look potentially non-linear.

# In[20]:


# set up preliminary data for data for fitting the models 
# the first column is the log median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston_input_df.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T


# In[21]:


# Splitting the dataset into the Training set and Test set
X = prelim_model_data[:,1:13]
y = prelim_model_data[:, 0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)


# Investigation Correlations and multicollinearity issues   
# 
# Correlation Matrix
# We are interested in keeping most of the indicators. If we were trying to infer which to keep, we would need to apply correlation only after splitting the data into a training and test set.

# In[22]:


# Get top absolute correlations
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = pd.DataFrame(X_train).columns
    for i in range(0, X_train.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = pd.DataFrame(X_train).corr(method='pearson').abs().unstack()
    labels_to_drop = get_redundant_pairs(X_train)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(X_train, 5))


# In[23]:


# Evaluate Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

prelim_model_data_c = add_constant(pd.DataFrame(prelim_model_data))
pd.Series([variance_inflation_factor(prelim_model_data_c.values, i) 
               for i in range(prelim_model_data_c.shape[1])], 
              index=prelim_model_data_c.columns)


# In[24]:


# Review the eigen values.
xs = X_train[:,1:13]    # independent variables
corr = np.corrcoef(xs, rowvar=0)  # correlation matrix
w, v = np.linalg.eig(corr)        # eigen values & eigen vectors
print(w) # Weighted distance from to employment center is near zero
v[:,7] # We see the issue is with number 8 and 9 which are rad and tax rate
v[:,6]


# Running the model without varaibles with a high VIF doesnt seem to improve the score. We will keep all variables.

# In[25]:


# Splitting the dataset into the Training set and Test set
X = prelim_model_data[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)


# In[26]:


# Scale Data. We fit to training and simply transform test. 
# While training should be similar to test data with large data- there could 
# still be differences and we must act as if we don't have access to test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler2 = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler2.fit_transform(y_train.reshape(-1, 1))
X_test = scaler.transform(X_test)
y_test = scaler2.transform(y_test.reshape(-1, 1))
model_data = np.concatenate((y_train, X_train), axis=1)


# In[27]:


# Define Ridge Regression
 
ridge_reg = Ridge(alpha=5, solver="cholesky", random_state=42)
ridge_reg.fit(X_train, y_train)


# In[28]:


# Tree-structured regression (simple)
# --------------------------------------
# try tree-structured regression on the original explantory variables
# note that one of the advantages of trees is no need for transformations
# of the explanatory variables... sklearn DecisionTreeRegressor
tree_model_maker = DecisionTreeRegressor(random_state = 15, max_depth = 5)
tree_model_fit = tree_model_maker.fit(X_train, y_train)


# In[29]:


# Random Forest and Extra trees regressor
extratrees = ExtraTreesRegressor(random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint


# In[30]:


# Let's tune the parameters for Ridge Regression
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'alpha': [.3, 5, 8,]},
  ]
grid_searchRR = GridSearchCV(ridge_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_searchRR.fit(X_train, y_train)

grid_searchRR.best_params_
grid_searchRR.best_estimator_


# In[31]:


# Let's tune the parameters for Decision Tree


param_grid = [
    {'max_depth': [4,6,8],
    'max_features': ['log2'],
     'min_samples_leaf': [1,2],
     'min_samples_split': [2,3]},
  ]

grid_search = GridSearchCV(tree_model_maker, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

grid_search.best_estimator_


# In[32]:


# Randomized search (could also use grid search) for Random Forest
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
{'bootstrap': [True, False],
 'max_depth': [50, 60, 70, 80, 90, 100, 200, 300, 500],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [1000, 1200, 1400]}


# In[33]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train.ravel())


# In[34]:


rf_random.best_params_


# In[35]:


extra_trees_reg = ExtraTreesRegressor(random_state=42)


# In[36]:


# Randomized search for Extra Trees
param_grid = [
    {'bootstrap': [True],
 'max_depth': [8, 10],
 'max_features': ['log2'],
 'min_samples_leaf': [1],
 'min_samples_split': [2],
 'n_estimators': [100, 200]}
,
  ]
grid_searchET = GridSearchCV(extra_trees_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_searchET.fit(X_train, y_train.ravel())

grid_searchET.best_estimator_


# In[37]:


#Feature Importance
#Fit model with optimal parameters on dataset
extra_trees_reg = RandomForestRegressor(random_state=42, bootstrap=True, max_depth = 18, min_samples_split = 2, n_jobs=-1, 
                               max_features='log2', n_estimators=1900)
extra_trees_reg.fit(X_train, y_train.ravel())
print("Feature Importances: {}".format(extra_trees_reg.feature_importances_))


# In[38]:


bag_reg = BaggingRegressor(
    DecisionTreeRegressor(random_state=42), n_estimators=1000,
    max_samples=300, bootstrap=True, n_jobs=-1, random_state=42)
bag_reg.fit(X_train, y_train.ravel()) 


# Comparing regressors with tuned paramters

# In[39]:


SET_FIT_INTERCEPT = True

names = ["LinearRegression", "Ridge","ExtraTreesRegressor", "RandomForestRegressor", "BaggingRegressor" ]

regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT), 
              Ridge(alpha = 5, solver = 'cholesky',  
                     normalize = False, 
                     random_state = RANDOM_SEED),
              ExtraTreesRegressor(bootstrap=True, criterion='mse', max_depth=10,
                      max_features='log2', max_leaf_nodes=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
                      random_state=42),
              DecisionTreeRegressor(max_depth=6, max_features='log2', min_samples_leaf=1,
                      min_samples_split=2),
              RandomForestRegressor(bootstrap=True, n_estimators = 1200, criterion='mse', 
                      min_samples_split = 2, min_samples_leaf = 1, max_depth=8,
                      max_features='log2'),
              BaggingRegressor(DecisionTreeRegressor(random_state=42), n_estimators=1200, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
                      
              ]


# In[40]:


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
cv_results = np.zeros((N_FOLDS, len(names)))


# In[41]:




kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')


# In[42]:


index_for_method = 0  # initialize
for name, reg_model in zip(names, regressors):
    print('\nRegression model evaluation for:', name)
    print('Scikit Learn method:', reg_model)
    reg_model.fit(X_train, y_train.ravel())  # fit on the train set for this fold
    
     # evaluate on the test set for this fold
    y_train_predict = reg_model.predict(X_train)
    print('Coefficient of determination (R-squared):',
          r2_score(y_train, y_train_predict))
    fold_method_result = sqrt(mean_squared_error(y_train, y_train_predict))
    print(reg_model.get_params(deep=True))
    print('Root mean-squared error:', fold_method_result)
    cv_results[index_for_fold, index_for_method] = fold_method_result
    index_for_method += 1
  
index_for_fold += 1


# In[43]:


cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names


# In[44]:


print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
print(cv_results_df.mean())  


# ####################### Performance on Test Set #######################

# In[45]:


# set up numpy array for storing results
names = ["LinearRegression", "Ridge","ExtraTreesRegressor", "RandomForestRegressor", "BaggingRegressor" ]
cv_results = np.zeros((N_FOLDS, len(names)))


# In[46]:


kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')


# In[47]:


# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for test_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')


    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('Scikit Learn method:', reg_model)
        reg_model.fit(X_test, y_test.ravel())  # fit on the test set for this fold
        
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


# In[48]:


cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names


# In[49]:


print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
print(cv_results_df.mean()) 


# Gradient Boosting

# In[50]:


from sklearn.metrics import mean_squared_error
gbrt = GradientBoostingRegressor(max_depth=4, n_estimators=100, random_state=42)
gbrt.fit(X_train, y_train.ravel())

errors = [mean_squared_error(y_test, y_pred.ravel())
          for y_pred in gbrt.staged_predict(X_test)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=4,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train.ravel())
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train.ravel())
    y_pred = gbrt.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping


# In[51]:


print(gbrt.n_estimators)


# In[52]:


print("Minimum validation MSE:", min_val_error)

