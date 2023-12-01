import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------- data preprocessing ---------- #

# load data
accidents_file_path = 'data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'
accidents_data = pd.read_csv(accidents_file_path)

# monat has form "202207"
accidents_data['MONAT'] = pd.to_datetime(accidents_data['MONAT'], format='%Y%m')
accidents_data.set_index('MONAT', inplace=True)

# save a copy to evaluate the performance.
initial_data = accidents_data.copy(deep=True)

# only use data before 2020 for developing the prediction model for 01/2021
accidents_data = accidents_data[accidents_data['JAHR'] < 2021]
# drop NA values
accidents_data = accidents_data.dropna(subset=['WERT'])

# filter to train on the relevant subset
accidents_data = accidents_data[(accidents_data['MONATSZAHL'] == 'Alkoholunfälle') &
                                (accidents_data['AUSPRAEGUNG'] == 'insgesamt')]

# chronological order
accidents_data.sort_index(inplace=True)

# visualize the data
plt.figure(figsize=(12, 6))
sns.set(style="darkgrid")
sns.lineplot(data=accidents_data, x='MONAT', y='WERT')
plt.xlabel('Date')
plt.ylabel('Amount of Accidents')
# plt.show()


# ---------- feature engineering ----------

# use lag features and rolling window for time series forecasting
month_window = 12  # set the month window for lag and window
for lag in range(1, month_window + 1):
    accidents_data[f'lag_{lag}'] = accidents_data['WERT'].shift(lag)

# apply rolling window to calculate mean over 12 months
accidents_data['rolling_mean'] = accidents_data['WERT'].rolling(window=month_window).mean()
# get rid of NaN values for training
accidents_data.dropna(inplace=True)

# set the features and target accordingly
features = [f'lag_{lag}' for lag in range(1, month_window + 1)] + ['rolling_mean']
X = accidents_data[features]
y = accidents_data['WERT']

# ---------- model training ----------

# split into train and test, do not shuffle to keep chronological order
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# create prediction features for 01/2021 by using the most recent entries in the last month_window
lag_features = accidents_data['WERT'].tail(month_window)
# reverse to avoid resemblance of most recent data
lag_features = lag_features.values[::-1]

# also add the last entries of the rolling_window
rolling_window_features = accidents_data['rolling_mean'].iloc[-1]

# create input feature array for the prediction
prediction_features = np.append(lag_features, rolling_window_features)
# reshape 1D to 2D array
prediction_features = prediction_features.reshape(1, -1)

y_pred = model.predict(prediction_features)

# ---------- performance evaluation ----------

# get the actual value from the initial data for 2021/01 following the alcohol and total criteria
actual = initial_data.loc[(initial_data['MONATSZAHL'] == 'Alkoholunfälle') &
                          (initial_data['AUSPRAEGUNG'] == 'insgesamt') &
                          (initial_data['JAHR'] == 2021)
                          ].loc['2021-01']['WERT'].iloc[0]

# use mse to compute the error between prediction and actual value
mse = mean_squared_error([y_pred[0]], [actual])

# ---------- results & visualization ----------

print("Forecast for total alcohol accidents in January 2021: {0}".format(y_pred[0]))
print('MSE for Linear Regression prediction for total alcohol accidents in January 2021: {0}'.format(mse))

plt.figure(figsize=(12, 6))
sns.lineplot(data=accidents_data, x='MONAT', y='WERT', label='Historical data of accidents')
january_2021 = mdates.date2num(pd.to_datetime('2021-01'))
plt.scatter(january_2021, actual, color='green', label='Actual for 01/2021', marker='o')
plt.scatter(january_2021, y_pred[0], color='orange', label='Prediction for 01/2021', marker='*')
plt.title('Forecast for total alcohol accidents in January 2021')
plt.xlabel('Date')
plt.ylabel('Amount of accidents')
plt.legend()
plt.grid(True)
plt.show()
