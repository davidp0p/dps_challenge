import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# ---------- data preprocessing ---------- #

# load data
accidents_file_path = 'data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'
accidents_data = pd.read_csv(accidents_file_path)

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
plt.figure(figsize=(15, 6))
sns.lineplot(data=accidents_data, x='MONAT', y='WERT')
plt.xlabel('Year')
plt.ylabel('Accidents')

plt.show()

print(accidents_data.describe())
print(accidents_data.info())



# ----- Model training -----