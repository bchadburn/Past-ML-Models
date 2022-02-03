# Cybersecurity Anomoly Detection 

![alt text](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/11/outlier.png)

The task was to detect potential malicious activity from log data on logins and requests. The data is unlabelled and 
requires unsupervised anomaly detection to identify potential threats.

### Cleaning
I took the following steps to clean and curate the data: 
* Removed columns with single value or with all unique values (except for 'id')
* Removed columns with less than 20% observations
* The removed columns previously mentioned were removed to disregard unhelpful for identifying anomalies
    - Those columns with few entries were manually reviewed avoid losing useful data but none were kept.
* Removing duplicated column data
* Clean column names - convert to snake case
* Manual reviewed remaining columns - dropped any other columns providing little to no insight

### Feature engineering
Created features based on recent activity grouped by user_id, user_access_key, and ip address. Tracked the following
activity:
* consecutive_failed_user_count: total failed requests since last successful request
* recent_failed_user_count: if user has recently failed a request (within a single minute)
* number_recent_user_outcomes: number of recent requests (within a day)
* recent_user_success: if user has recently made a successful request (within the last day)
* ip_user_counts: Indicator if requests had previously been made by that user from same ip address
* ip_user_recently_used: user recently used ip address

##### Binning
Histograms of high count columns were reviewed to determine appropriate bins.  
Binning was used to create better groupings, help with model performance, and to 
avoid creating unuseful outliers. For example, a user reaching 1200 requests may 
be uncommon, but we don't want to flag it as anomalous based on one feature. 
We also don't have 100 different values as we want to simplify the search space. 

#### Dummy variables
Explored
* successful request (1 if error_code = 'success' 0 if not)
* object_category
* user_identity_user_name

Final
* object_category

#### Explored transformations and scaling
* Using log transform for heavily skewed data
* Standardized with MinMaxScaler

Determined not to proceed with either as binning was used in hope to reduce data complexity. However, 
MDL-based binning, logarithmic binning, or SAX binning techniques could also be evaluated.  

### Algorithms used
* Isolation Forest
* OneClassSVM
* PCA for exploration - reviewing data separation

The Isolation Forest algorithm was used as it fits our data well since it includes a contamination factor, and we 
were given the information that our data is expected to have around .01% malicious activity. The algorithm 
was also used as it's capable of handling fairly complex behavior between multiple features. 

The OneClassSVM algorithm was chosen as its particularly well suited for binary classifications and includes an 'nu' 
parameter which gives us the expected ratio of outliers i.e. the fraction of support vectors.

#### Model Performance
* Report number of anomaly predictions vs expected
* Print PCA clustering in 2 and 3 dimensions to get some indications of data separation
* Return observations and manually review to see features being used and why it may qualify as an anomaly and if there's
any indication of suspicious activity

#### Results
The Isolation Forest returned 6 predictions with a contamination factor of .001, instead the factor must 
be increased, and then model returns many false positives. When we review the anomalies, we see the data points are quite
similar to other observations and lie just outside the decision boundaries.

The OneClassSVM does return a couple anomalies with a .001 'nu' value and that can be slightly increased to return 5+ observations
(our target is 5-6). A review of the 'anomalies' further show the observations are not drastically different from 
others and don't appear to be highly related to security threats. 

A grid search was also used to explore how the number of anomalies changed with changes of nu and gamma. However, 
as we don't have any labelled anomalies, there was little insight into the quality of anomalies in order to further tune
the model. 

PCA plots were created projecting the data into 2 and 3 dimensional representations. Similar to what we saw when reviewing 
the observations, both models indicated low data separation in the plots.

#### Improvements and next steps
The features I've explored so far provide some insight into request behavior. However, we aren't seeing much 
anomalous behavior even when comparing the requests by user access id and user id. With more time, I'd probably 
start with reviewing very specific activity and trying to get a feel for what typical behavior should look like 
surrounding specific events. This would help direct the next steps of improving our feature engineering.

I could also further explore specific user behavior and being able to compare current behavior with the typical user's 
past behavior. For instance, if we see usage from a different source, an untypical amount of activity, type of requests etc. 

The other feature improvement I'd try is to categorize certain requests into security threats vs benign activity. 
Doing so could help better focus on the type of behavior that's useful to monitor surrounds an activity. For instance, 
it may be uncommon for a user to have no recent activity and then start activity by immediately requesting credentials. 

I would also add two additional steps to evaluate the model's performance once I identified better features. I would
feed the model synthetic anomalous behavior, feeding extreme values for features we'd want the model to flag. The 
samples could also be simulated based on expected historic variance to get some idea of the probability of seeing the 
observations. Second, I would also swap user's behavior (particular two dissimilar users e.g., a frequent user and 
an infrequent user) and ensure the model would flag it as abnormal. If the features have adequately represented 
typical past user behavior it should be able to flag when the current user's behavior is abnormal 
(since we are actually feeding it behavior from another user).

Finally, improvement could be made comparing and tuning other anomaly detection models once the variables are improved. 
It would likely be useful to try modeling with subsets of the features to better find anomalies for 
certain types of behavior or activity. At any rate, designing better engineering features which aggregate past data 
and capture particularly insightful behavior would be a necessary first step before moving on to stacking or ensemble 
type modeling efforts. 
