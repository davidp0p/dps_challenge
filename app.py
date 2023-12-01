import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------- data preprocessing ---------- #

# load data
accidents_file_path = 'data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'
accidents_data = pd.read_csv(accidents_file_path)
raw_data = pd.read_csv(accidents_file_path)

# monat has form "202207"
accidents_data['MONAT'] = pd.to_datetime(accidents_data['MONAT'], format='%Y%m')
accidents_data.set_index('MONAT', inplace=True)

# only use data before 2020 for developing the prediction model
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
plt.xlabel('Year')
plt.ylabel('Accidents')
#plt.show()


# ---------- feature engineering ----------

# use lag features and rolling window for time series forecasting
month_window = 12  # set the month window for lag and window
for lag in range(1, month_window + 1):
    accidents_data[f'lag_{lag}'] = accidents_data['WERT'].shift(lag)

# apply rolling window to calculate mean over 12 months
accidents_data['rolling_window'] = accidents_data['WERT'].rolling(window=month_window).mean()

accidents_data.dropna(inplace=True)

# set the features and target accordingly
features = [f'lag_{lag}' for lag in range(1, month_window + 1)] + ['rolling_window']
X = accidents_data[features]

y = accidents_data['WERT']


# ---------- model training ----------

# split into train and test, do not shuffle to keep chronological order
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)


# create features for prediction
lag_features = accidents_data['WERT'][-month_window:].values[::-1]
rolling_window_features = accidents_data['rolling_window'].iloc[-1]
prediction_features = np.append(lag_features, rolling_window_features)
# reshape 1D to 2D array
prediction_features = prediction_features.reshape(1, -1)

y_pred = model.predict(prediction_features)


