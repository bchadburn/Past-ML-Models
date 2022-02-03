
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


class ValueTooSmall(Exception):
    pass

def get_start_dates(data, days_eval, days_baseline):
    start_date = (max(data['time']) - pd.Timedelta(days=days_eval)).floor('D')
    if start_date < min(data['time']):
        raise ValueTooSmall("Time span of web traffic precedes data")
    baseline_start = start_date - pd.Timedelta(days=days_baseline)
    if start_date < min(data['time']):
        raise ValueTooSmall("Time span used to calculate baseline precedes data")
    return start_date, baseline_start


def check_matching_days(data, col):
    if data[col].isna().sum() > 0:
        warnings.warn(f"Some days have null values. Plots may be inaccurate")


# Time span, in days, we want to calculate metrics
time_span_eval = 7
time_span_baseline = 14  # The dashboard currently uses same number of days, however, this supports different ranges

# Import data
DIRECTORY = 'C:\\Users\\Windows\\Desktop\\Projects\\Career\\Tatari'
web_traffic = pd.read_csv(f'{DIRECTORY}\\assignment-analyst-1-web-traffic-data.csv')
spot_data = pd.read_csv(f'{DIRECTORY}\\assignment-analyst-1-spot-data.csv')

# Clean data
web_traffic = web_traffic[web_traffic['value'] >= 0]  # Remove negative web traffic values
web_traffic['time'] = pd.to_datetime(web_traffic['time'], utc=True)
spot_data['time'] = pd.to_datetime(spot_data['time'], utc=True)
web_traffic['time_day'] = pd.to_datetime(web_traffic['time'].dt.strftime('%Y-%m-%d'))
spot_data['time_day'] = pd.to_datetime(spot_data['time'].dt.strftime('%Y-%m-%d'))
web_traffic['time_hr'] = pd.to_datetime(web_traffic['time'].dt.strftime('%Y-%m-%d-%H'))
spot_data['time_hr'] = pd.to_datetime(spot_data['time'].dt.strftime('%Y-%m-%d-%H'))
web_traffic['day_of_week'] = web_traffic['time'].dt.weekday
spot_data['day_of_week'] = spot_data['time'].dt.weekday

# Subset data
end_date = max(web_traffic['time']).ceil('D') - pd.Timedelta(days=1)  # Returns last complete day recorded
starting_date, baseline_start_date = get_start_dates(web_traffic, time_span_eval, time_span_baseline)

comparison_visits_data = web_traffic[(web_traffic['time'] <= starting_date) & (web_traffic['time'] >= baseline_start_date)]
comparison_spot_data = spot_data[(spot_data['time'] <= starting_date) & (spot_data['time'] >= baseline_start_date)]

recent_visits_data = web_traffic[(web_traffic['time'] > starting_date) & (web_traffic['time'] < end_date)]
recent_spot_data = spot_data[(spot_data['time'] > starting_date) & (spot_data['time'] < end_date)]

# Calculate baseline
anchor_day_hr_visits_data = comparison_visits_data.groupby(['time_hr'], as_index=False)[['value']].sum().round(2)
anchor_day_hr_spot_data = comparison_spot_data.groupby(['time_hr'], as_index=False)[['spend']].sum().round(2)
anchor_day_hr_visits_data = pd.merge(anchor_day_hr_visits_data, anchor_day_hr_spot_data, how='left', on=['time_hr'])

anchor_day_hr_visits_data.loc[~anchor_day_hr_visits_data.spend.isna(), 'value'] = np.nan
anchor_day_hr_visits_data = anchor_day_hr_visits_data.ffill().bfill()  # Todo: impute by predicting value based on time of day.
anchor_day_hr_visits_data['hr'] = anchor_day_hr_visits_data['time_hr'].dt.hour
anchor_day_hr_visits_data['day_of_week'] = anchor_day_hr_visits_data['time_hr'].dt.weekday

hr_count = len(anchor_day_hr_visits_data)
daily_baseline = round((np.sum(anchor_day_hr_visits_data['value'])/hr_count) * 24, 2)
print('Daily baseline:', daily_baseline)

# Calculate comparison
seasonal_component = 0  # Placeholder for adjusting for seasonality. We won't incorporate these for this assignment
trend_component = 0  # Place holder for trend component
baseline_visitors = daily_baseline * (1 + seasonal_component + trend_component)

# Group past and current visits by weekday
anchor_weekday_visitors = anchor_day_hr_visits_data.groupby(['day_of_week'], as_index=False)[['value']].sum().round(2)
comparison_weekday_visits_data = comparison_visits_data.groupby(['day_of_week'], as_index=False)[['value']].sum().round(2)
comparison_weekday_visits_data = pd.merge(comparison_weekday_visits_data, anchor_weekday_visitors, how='left', on=['day_of_week'],
                                     suffixes=('', '_baseline'))
comparison_unique_hr_cnt = comparison_visits_data.drop_duplicates(subset='time_day')
comparison_daily_cnt = comparison_unique_hr_cnt.groupby(['day_of_week'], as_index=False)[['value']].count().round(2)
comparison_weekday_visits_data = pd.merge(comparison_weekday_visits_data, comparison_daily_cnt, how='left', on=['day_of_week'],
                                     suffixes=('', '_cnt'))
comparison_weekday_visits_data['value_comparison'] = comparison_weekday_visits_data['value'] / comparison_weekday_visits_data['value_cnt']
comparison_weekday_visits_data.drop(columns=['value', 'value_cnt'], inplace=True)

recent_visits_tbl = recent_visits_data.groupby(['time_day', 'day_of_week'], as_index=False)[['value']].sum().round(2)
recent_visits_tbl = pd.merge(recent_visits_tbl, comparison_weekday_visits_data, how='left', on=['day_of_week'],
                             suffixes=('', '_comparison'))
recent_visits_tbl['lift'] = (recent_visits_tbl['value'] - recent_visits_tbl['value_baseline'])
recent_visits_tbl['lift_percent'] = (recent_visits_tbl['value'] / recent_visits_tbl['value_baseline']) - 1

# Overall lift
recent_visitors = np.sum(recent_visits_tbl['value'])
lift = round(recent_visitors - np.sum(recent_visits_tbl['value_baseline']), 2)
lift_percent = recent_visitors / np.sum(recent_visits_tbl['value_baseline']) - 1
print('Total lift:', lift)
print('Overall lift as % of baseline: ', '{:.2%}'.format(lift_percent))

check_matching_days(recent_visits_tbl, 'lift')
plt.figure()
plt.title('Lift Over Time')
plt.ylabel('Lift')
plt.plot(recent_visits_tbl['time_day'], recent_visits_tbl['lift'])
plt.gcf().autofmt_xdate()
plt.savefig('Lift Over Time.png')
plt.show()

# Compare website visits overtime
day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
recent_visits_tbl['weekday'] = recent_visits_tbl['day_of_week'].map(day_map)

plt.figure()
plt.title('Compare Visit Over Time')
plt.ylabel('Visits')
plt.plot(recent_visits_tbl['weekday'], recent_visits_tbl['value_comparison'], label='Past Visits')
plt.plot(recent_visits_tbl['weekday'], recent_visits_tbl['value'], label='Current Visits')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('Visit Comparison Over Time.png')
plt.show()

# Spend per visit
mean_spend_per_visit = np.sum(recent_spot_data['spend']) / lift
print('Overall spend:', round(mean_spend_per_visit, 2))

recent_spot_tbl = recent_spot_data.groupby(['time_day', 'day_of_week'], as_index=False)[['spend']].sum().round(2)
recent_visits_tbl = pd.merge(recent_visits_tbl, recent_spot_tbl, how='left', on=['day_of_week'], suffixes=('', '_comparison'))
recent_visits_tbl['lift'] = recent_visits_tbl['value'] - recent_visits_tbl['value_baseline']
recent_visits_tbl['spend_per_visit'] = recent_visits_tbl['spend'] / recent_visits_tbl['lift']

plt.figure()
plt.title('Spend Per Visit')
plt.ylabel('Avg Spend')
plt.plot(recent_visits_tbl['time_day'], recent_visits_tbl['spend_per_visit'])
plt.gcf().autofmt_xdate()
plt.savefig('Spend per Visit.png')
plt.show()
