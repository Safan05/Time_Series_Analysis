import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Load time series data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)
demand = data['Electric_demand']  # Replace 'Electric_demand' with your column name for electricity demand
temperature = data['Temperature']  # Replace 'Temperature' with your column name for temperature

# Split data into training and testing sets
train_data = data[:'2020']
test_data = data['2021':]

train_demand = train_data['Electric_demand']
test_demand = test_data['Electric_demand']
test_temperature = test_data['Temperature']

# Make the series stationary (if needed)
result = adfuller(train_demand)
if result[1] > 0.05:  # If p-value > 0.05, apply differencing
    train_demand_diff = train_demand.diff().dropna()
    train_demand = train_demand_diff

# Determine ARIMA (p, d, q) using PACF and ACF plots
plot_acf(train_demand)
plt.savefig('acf_plot.png')
plt.close()

plot_pacf(train_demand)
plt.savefig('pacf_plot.png')
plt.close()

# Fit ARIMA model
model = ARIMA(train_demand, order=(1, 1, 1))  # Replace with appropriate (p, d, q)
model_fit = model.fit()

# Forecast for 2021
forecast_steps = len(test_data)  # Number of steps to forecast
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Save the forecast and confidence intervals to a CSV file
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean,
    'Lower CI': confidence_intervals.iloc[:, 0],
    'Upper CI': confidence_intervals.iloc[:, 1]
}, index=test_data.index)
forecast_df.to_csv('2021_forecast_with_ci.csv')

# Plot forecast with confidence intervals
plt.figure(figsize=(10, 5))
plt.plot(train_demand, label="Observed (Train)")
plt.plot(test_data.index, forecast_mean, label="Forecast (2021)", color='red')
plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.title("Electricity Demand Forecast with Confidence Intervals")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.savefig('forecast_plot_with_ci.png')
plt.close()

# Calculate mean squared error
mse = mean_squared_error(test_demand, forecast_mean)
print(f'Mean Squared Error: {mse}')

# Outlier analysis on temperature
temperature_mean = test_temperature.mean()
temperature_std = test_temperature.std()
outliers = test_temperature[(test_temperature < temperature_mean - 3 * temperature_std) | (test_temperature > temperature_mean + 3 * temperature_std)]

# Examine if anomalies coincide with significant deviations in forecast accuracy
forecast_errors = test_demand - forecast_mean
significant_deviations = forecast_errors[(forecast_errors > forecast_errors.mean() + 3 * forecast_errors.std()) | (forecast_errors < forecast_errors.mean() - 3 * forecast_errors.std())]

# Print outliers and significant deviations
print("Temperature Outliers:")
print(outliers)
print("Significant Forecast Deviations:")
print(significant_deviations)