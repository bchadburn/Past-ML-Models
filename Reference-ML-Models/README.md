Overview of model scripts

The folder "Linear Regression" includes several different type of linear regressions conducted in R programming. These include GLM, Logistic Regression, "Predicting House Prices" using GLM but with 80 variables to sort through, Regression with PCA transformation, and the model "poisson_negative_binomial_ZIP". Most of these are self-explanatory. The last one is for models trying to predict values where zero values are common and so Poisson, Negative Binomial, as well as a Zero Inflated Negative Binomial model was used to compare model performances.

The folder NLP-Topic-Modeling is a project for creating a taxonomy of job titles (in this example we used "data scientist", but it could easily apply to any title) and to determine potential equivalent classes, similar titles and sub-domains. For more details, see the README in the NLP-Topic-Modeling folder.

"Chicago_Food_Inspection" (Python) uses elastic search to query the chicago food inspection data set using NoSQL database engine ElasticSearch for indiexing and data retrieval. Use heatmap to plot the geolocations of the facilities that failed Chicago food inspections.

"Random_Forests" predicts Boston Housing prices using various random foreest algorithms. Extra trees regressor, Ridge Regression, linear regression, Gradient Boosting, and Bagging Regressor are all used to make predictions and then using k-fold cross-validation their performance was compared using mean squared error (MSE). 

"NLP_RNN" (Python) is a recurrent neural network (RNN) using TensorFlow for natural language proceesing (NLP). The model uses GloVe to represent words as neural network embeddings to assist with deriving word meaning and use. This is the one assignment provided on github, where the template code makes up the majority of the code as the process is quite lengthly. However, I was required created two alternative RNN structures and performed hyperparameter testing. Using starter code as template, I then created four seperate
language models.

The file "logistic classifier bank" (Python) uses naive bayes and logistic regression classifiers and uses k-fold cross validation.

The folder "NLP Classification" was a project for predicting the job discipline (e.g. data scientist vs data engineer) using job titles and job descriptions from Indeed. This required using labelled data to train and test the model. It includes incorporating extensive skill categories information to provide greater insight into the model, using ensemble methods, and stacking. 
