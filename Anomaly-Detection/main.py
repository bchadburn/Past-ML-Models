
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


def clean_column_names(col_names):
    try:
        col_names = list(col_names)
    except TypeError:
        print("Must be passed np.array or list")
    col_cleaned = []
    for name in col_names:
        tmp_name = re.sub(r'[^\w\s]', '_', name)  # Replace punctuation with underscore
        if tmp_name[0] == '_':
            tmp_name = tmp_name.split('_')[1]
        tmp_name = tmp_name.replace('ID', 'id')
        tmp_name = tmp_name.replace('IP', 'ip')
        tmp_name = re.sub(r'(?<!^)(?=[A-Z])', '_', tmp_name).lower()  # Camel case to snake case
        col_cleaned.append(tmp_name)
    return col_cleaned


def plot_pca_transform(data, anomaly_idx, components):
    """Conduct PCA dimensionality reduction and plot in 2 or 3 dimensions"""
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(data.values)
    fig = plt.figure()

    if components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("pca_composite")
        ax.scatter(pca_result[:, 0], pca_result[:, 1], zs=pca_result[:, 2], s=4, label="normal", c="green")
        ax.scatter(pca_result[anomaly_idx, 0], pca_result[anomaly_idx, 1], pca_result[anomaly_idx, 2],
                   lw=3, s=50, marker="x", c="red", label="outlier")
    if components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(pca_result[:, 0], pca_result[:, 1], s=4, label="normal", c="green")
        ax.scatter(pca_result[anomaly_idx, 0], pca_result[anomaly_idx, 1],
                   lw=3, s=50, marker="x", c="red", label="outlier")
    ax.set_title('PCA plot of observations')
    ax.legend()
    plt.show()


def return_failed_count(outcomes, current_time, start_time_list, current_outcome):
    """Return features for recently failed requests"""
    consecutive_failed_count, failed_count_same_minute, failed_count, prev_outcome = 0, 0, 0, 0
    consecutive_flag = True
    if current_outcome != 'success':
        for outcome, _time in zip(outcomes, start_time_list):
            if outcome != 'success':
                if consecutive_flag:
                    consecutive_failed_count += 1
                    if (current_time - _time).total_seconds() / 60 < 1:
                        failed_count_same_minute += 1
                    else:
                        failed_count_same_minute = 0
            if outcome == 'success':
                consecutive_flag = False
    return consecutive_failed_count, failed_count_same_minute


def bin_variable(key, bins):
    """Bin variable"""
    for _bin in bins:
        if key > _bin:
            continue
        if key == _bin:
            return _bin
        else:
            return bins[bins.index(_bin) - 1]


def return_past_outcomes(outcomes, current_time, start_times_list):
    """Return features for past successful requests"""
    recent_success = 0
    num_recent_outcomes = 0
    if 'success' in outcomes:
        successful_count = 0
        for outcome, _time in zip(outcomes, start_times_list):
            if outcome == 'success':
                successful_count += 1
                if (current_time - _time).total_seconds() / 60 / 24 < 1:
                    recent_success += 1
            elif (current_time - _time).total_seconds() / 60 / 24 < 1:
                num_recent_outcomes += 1
    else:
        recent_success = 0
    return recent_success, num_recent_outcomes


def return_ip_activity(data, time_col, ip_col, ip_current, current_time):
    """Return features based on past activity from same IP address"""
    ip_count = len(data[data[ip_col] == ip_current])
    past_day = (current_time - timedelta(days=1))
    if len(data[(data[ip_col] == ip_current) & (data[time_col] > past_day)]) > 0:
        ip_recently_used = 1
    else:
        ip_recently_used = 0
    return ip_count, ip_recently_used


def compute_recent_attempts(data, col, time_col):
    """Return features for recent failed/successful attempts. Also return features regarding a user
    sending requests from different ip addresses, has user made that request before."""
    attempts_dict = {'consecutive_failed_count': [], 'recent_failed_count': [], 'failed_counts': [],
                     'number_successful_attempts': [], 'previous_success': [], 'recent_success': [],
                     'number_recent_outcomes': [], 'ip_counts': [], 'ip_recently_used': []}
    data.sort_values(by=[col, time_col], inplace=True, ascending=True)
    data.reset_index(drop=True, inplace=True)
    for index, row in data.iterrows():
        if index % 500 == 0:
            print(index)
        tmp_data = data[0:index]
        tmp_data = tmp_data[tmp_data[col] == row[col]]

        current_outcome = row['error_code']
        recent_outcomes = tmp_data['error_code'].values.tolist()
        recent_outcomes.reverse()
        past_dates = tmp_data['start_time'].values.tolist()
        past_dates = [pd.to_datetime(t) for t in past_dates]
        past_dates.reverse()

        recent_success_count, num_outcomes = return_past_outcomes(recent_outcomes, row['start_time'], past_dates)
        consecutive_failed_count, recent_failed_count = return_failed_count(recent_outcomes, row['start_time'],
                                                                            past_dates, current_outcome)
        ip_count, ip_recently_used = return_ip_activity(tmp_data, 'start_time', 'src_ip', row['src_ip'], row['start_time'])

        attempts_dict['consecutive_failed_count'].append(consecutive_failed_count)
        attempts_dict['recent_failed_count'].append(recent_failed_count)
        attempts_dict['recent_success'].append(recent_success_count)
        attempts_dict['number_recent_outcomes'].append(num_outcomes)
        attempts_dict['ip_counts'].append(ip_count)
        attempts_dict['ip_recently_used'].append(ip_recently_used)
    return attempts_dict


def create_histograms(data, col_names, limit=False, grouped_by=False):
    for col_name in col_names:
        if not grouped_by:
            tbl = data[col_name]
        else:
            tbl = data.groupby([grouped_by], as_index=False)[[col_name]].sum().round(2)
        if not limit:
            pass
        else:
            tbl = tbl[tbl[col_name] <= limit]
        plt.figure()
        plt.title(col_name)
        tbl[col_name].plot(kind='hist', bins=10)
        plt.show()


def add_dummy_columns(data, feature_col):
    dummy_tbl = pd.get_dummies(data[[feature_col]])
    final_tbl = pd.concat([data, dummy_tbl], axis=1)
    return final_tbl


def log_transform_columns(data, _columns):
    constant = .00001  # Set constant to transform zero value observations
    for col in _columns:
        cleaned_logdata[col] = cleaned_logdata[col] + constant
        cleaned_logdata[col] = np.log(cleaned_logdata[col])
    return data


def bin_columns(data, col_list, buckets):
    for col in col_list:
        data[col] = [bin_variable(x, buckets) for x in data[col]]
    return data


def plot_count_by_category(data):
    for x in data.columns[data.dtypes == object]:
        fig = plt.figure()
        data[x].value_counts(normalize=True).plot(kind='bar')
        fig.suptitle(x)
        plt.show()


def fit_predict_model(model, data):
    model.fit(data.values)
    print(model.get_params())
    predictions = model.predict(data.values)
    print('# of anomalies predicted:', (predictions == -1).sum())
    return predictions


def run_grid_search(data, model, grid_parameters):
    for p in ParameterGrid(grid_parameters):
        print()
        print('Parameters:', model.set_params(**p))
        predictions = fit_predict_model(model, data)
        print('# of anomalies predicted:', (predictions == -1).sum())


def evaluate_anomalies(data, preds):
    data['anomaly'] = preds
    anomaly = data.loc[data['anomaly'] == -1]
    anomaly_index = list(anomaly.index)

    print("# of Outliers Found:", list(data['anomaly']).count(-1))
    print("# of Expected Outliers:", len(data) * .001)

    if len(anomaly) > 0:
        plot_pca_transform(data, anomaly_index, 3)  # Reduce to 3 dimensions
        plot_pca_transform(data, anomaly_index, 2)
    return anomaly_index


# Load data
logdata = pd.read_csv('C:\\Users\\Windows\\Desktop\\Projects\\ReliaQuest\\take_home_data_science\\logdata.csv')
logdata = logdata.dropna(thresh=len(logdata) * .2, axis=1)  # Remove columns with < 20% observations

# Clean data
start_times = [datetime.fromisoformat(date[:-1]).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for date in logdata['start_time']]
logdata['start_time'] = pd.to_datetime(start_times)

logdata = logdata[[col for col in logdata if logdata[col].nunique() > 1]]  # Remove columns with single values or all unique
cleaned_logdata = logdata[[col for col in logdata if logdata[col].nunique() != len(logdata)]]
cleaned_logdata.loc[:, 'id'] = logdata.loc[:, 'id']  # Add back id

clean_col_names = clean_column_names(cleaned_logdata.columns.values)
cleaned_logdata.columns = clean_col_names
cleaned_logdata = cleaned_logdata.loc[:, ~cleaned_logdata.columns.duplicated()]
cleaned_logdata.rename(columns={'eventtype': 'event_type_cloud_trail'}, inplace=True)

# Remove duplicated and other unused columns
cleaned_logdata.drop(columns=['bkt', 'date_hour', 'dest', 'command', 'date_minute', 'app', 'date_second', 'indextime',
                              'time', 'tag', 'aws_region', 'punct', 'region', 'source', 'sourceip_address', 'src',
                              'user_identity_arn', 'response_elements_config_rules_evaluation_status___config_rule_arn',
                              'response_elements_config_rules_evaluation_status___config_rule_id',
                              'user_identity_session_context_attributes_creation_date',
                              'response_elements_config_rules_evaluation_status___last_successful_evaluation_time',
                              'response_elements_config_rules_evaluation_status___last_successful_invocation_time',
                              'response_elements_config_rules_evaluation_status___config_rule_name', 'tag__eventtype',
                              'signature', 'user_identity_principal_id', 'user_name', 'dvc', 'request_parameters_config_rule_names__',
                              'user', 'user_arn', 'vendor_region', 'msg', 'user_identity_access_key_id', 'event_time',
                              'timeendpos', 'timestartpos', 'event_version'], inplace=True)

# Create recent activity features
user_attempts_dict = compute_recent_attempts(cleaned_logdata, 'user_access_key', 'start_time')
cleaned_logdata['consecutive_failed_user_count'] = user_attempts_dict['consecutive_failed_count']
cleaned_logdata['recent_failed_user_count'] = user_attempts_dict['recent_failed_count']
cleaned_logdata['recent_user_success'] = user_attempts_dict['recent_success']
cleaned_logdata['number_recent_user_outcomes'] = user_attempts_dict['number_recent_outcomes']
cleaned_logdata['ip_user_counts'] = user_attempts_dict['ip_counts']
cleaned_logdata['ip_user_recently_used'] = user_attempts_dict['ip_recently_used']

# user_attempts_dict = compute_recent_attempts(cleaned_logdata, 'user_id', 'start_time')
# cleaned_logdata = pd.read_csv('C:\\Users\\Windows\\Desktop\\subset_logdata.csv')

# Review distribution of high count columns
high_cnt_columns = ['number_recent_user_outcomes', 'ip_user_counts']

# Review distribution of data
create_histograms(cleaned_logdata, high_cnt_columns, 1000, grouped_by='user_access_key')

bins = [0, 5, 10, 50, 100, 500, 1000, 5000, 50000, 100000]
cleaned_logdata = bin_columns(cleaned_logdata, ['number_recent_user_outcomes', 'ip_user_counts', 'recent_user_success',
                                                'consecutive_failed_user_count', 'recent_failed_user_count'], bins)

# # Log transform columns
# cleaned_logdata = log_transform_columns(cleaned_logdata, high_cnt_columns)

# # Standardize columns to be on same scale as other variables
# scaler = MinMaxScaler()
# cleaned_logdata[high_cnt_columns] = scaler.fit_transform(cleaned_logdata[high_cnt_columns])

# Review categorical columns
columns = ['error_code', 'user_identity_user_name']
categorical_data = cleaned_logdata[columns]

# Print value counts by category
plot_count_by_category(categorical_data)

# Convert error code to binary - successful/not successful
cleaned_logdata['success'] = 0
cleaned_logdata.loc[cleaned_logdata['error_code'] == 'success', 'success'] = 1

# Combine columns as needed
cleaned_logdata.loc[~cleaned_logdata['object_category'].isin(['unknown', 'instance']), 'object_category'] = 'object_category_other'


# Create dummy columns
cleaned_logdata = add_dummy_columns(cleaned_logdata, 'object_category')

# Drop columns we aren't using
logdata_features = cleaned_logdata.copy()

original_columns = ['error_code', 'object_category', 'user_identity_user_name', 'start_time']
logdata_features.drop(columns=original_columns, inplace=True)

unused_columns = ['event_name', 'requestid', 'user_access_key', 'change_type', 'src_ip', 'user_agent',
                  'event_type_cloud_trail', 'event_type', 'user_id', 'event_source']
logdata_features.drop(columns=unused_columns, inplace=True)

unbinned_columns = ['number_recent_user_outcomes', 'ip_user_counts']
logdata_features.drop(columns=unbinned_columns, inplace=True)

# Create models
data_vars = logdata_features.iloc[:, 1:]  # Drops id column for modeling
iso_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.005, random_state=99)
svm = OneClassSVM(nu=.001, gamma='scale')

# Fit models
iso_predictions = fit_predict_model(iso_model, data_vars)
svm_predictions = fit_predict_model(svm, data_vars)

# Grid search
grid = {'gamma': ['scale'], 'nu': [0.001, .01]}
run_grid_search(data_vars, svm, grid)

# Return Anomalies
iso_anomoly_index = evaluate_anomalies(data_vars, iso_predictions)
svm_anomoly_index = evaluate_anomalies(data_vars, svm_predictions)
print(svm_anomoly_index)
