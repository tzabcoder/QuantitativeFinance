# File imports
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Package settings
plt.style.use('ggplot')

# Data name mapping (filename, column name)
# Data was downloaded from FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org/
file_names = {
    'data/energy_data/All_Employees_Oil_Gas_Extraction.csv': 'employees',
    'data/energy_data/City_Average_Price_Per_kWH.csv': 'city_average',
    'data/energy_data/Industrial_Production_Utilities_Electric_Gas.csv': 'production',
    'data/energy_data/Manufacturer_New_Orders_Lighting_Equipment.csv': 'orders_lighting',
    'data/energy_data/Manufacturer_New_Orders_Mining_Oil_Gas.csv': 'orders_mining_oil_gas',
    'data/energy_data/WTI_Oil_Prices.csv' : 'oil_price'
}

# Read and combine all files using list comprehension and concat
data = pd.concat([
    pd.read_csv(f).rename(columns={
        'observation_date' : 'observation_date', # Keep the same date column
        f : file_names[f]                        # Rename the column
    }).set_index('observation_date')             # Set the date column as index
    for f in file_names.keys()
], axis=1)

# Clean and adjust the data
data = data.astype(float)
data.index = pd.to_datetime(data.index)
data.drop(data.index[-1], inplace=True)

# Data Visualization ==================================================
# Time series plots
N_ROWS, N_COLS = 6, 1
fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(20, 20), sharex=True)
fig.suptitle('Energy Sector Data')

idx = 0
for k, v in file_names.items():
    axes[idx].plot(data[v])
    axes[idx].set_ylabel(v, rotation=0, fontsize=10, labelpad=40)
    idx += 1
plt.show()

# Distribution plots
N_ROWS, N_COLS, N_BINS = 3, 2, int(math.sqrt(len(data)))
fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(20, 20))

r_idx = 0
c_idx = 0
for k, v in file_names.items():
    sns.histplot(data[v], bins=N_BINS, kde=True, ax=axes[r_idx][c_idx])
    axes[r_idx][c_idx].set_title(v)
    axes[r_idx][c_idx].set_xlabel(v, labelpad=40)
    axes[r_idx][c_idx].set_ylabel('Frequeny', labelpad=40)

    c_idx += 1
    if c_idx == N_COLS:
        c_idx = 0
        r_idx += 1
plt.show()

# Autocorrelation plots
N_ROWS, N_COLS = 3, 2

# Lags based on data length
N_LAGS = min(40, len(data) // 5)

fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS)

r_idx = 0
c_idx = 0
for k, v in file_names.items():
    plot_acf(data[v], ax=axes[r_idx][c_idx], title=f"{v} ACF", lags=N_LAGS)
    axes[r_idx][c_idx].figure.set_size_inches(20, 20)

    c_idx += 1
    if c_idx == N_COLS:
        c_idx = 0
        r_idx += 1
plt.show()

# Partial autocorrelation plots
N_ROWS, N_COLS = 3, 2
fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS)

r_idx = 0
c_idx = 0
for k, v in file_names.items():
    plot_pacf(data[v], ax=axes[r_idx][c_idx], title=f"{v} PACF", lags=N_LAGS)
    axes[r_idx][c_idx].figure.set_size_inches(20, 20)

    c_idx += 1
    if c_idx == N_COLS:
        c_idx = 0
        r_idx += 1
plt.show()

# Scatter plot matrix
pd.plotting.scatter_matrix(data, figsize=(20, 20), diagonal='kde')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Testing for autocorrelation ==================================================
# For each column in the data, calculate the Ljung-Box test statistic and p-value
# Null hypothesis: The data is independently distributed (no autocorrelation)
# Alternative hypothesis: The data is not independently distributed (there is autocorrelation)
# If the p-value is less than 0.05, we reject the null hypothesis and conclude that there is autocorrelation in the data
P_VALUE_THRESHOLD = 0.05
P_VALUE_COL = "lb_pvalue"

for col in data.columns:
    lb_test = acorr_ljungbox(data[col], lags=N_LAGS, boxpierce=True)

    # Determine if we reject the null hypothesis
    lb_test['reject_null'] = np.where(lb_test[P_VALUE_COL] < P_VALUE_THRESHOLD, True, False)

    # Determine if there are any non-rejections
    if lb_test['reject_null'].any() is False:
        print(f"Test Results ({col}): \n{lb_test}")
    else:
        print(f"Reject the null hypothesis for {col}: autocorrelation")