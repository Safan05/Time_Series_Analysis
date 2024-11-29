import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load time series data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)
demand = data['Electric_demand']  # Replace 'Demand' with your column name for electricity demand
temperature = data['Temperature']  # Replace 'Temperature' with your column name for temperature
days = data.index.dayofweek  # Extract day of the week from the date index

# Add a column for the season
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data['Season'] = data.index.map(get_season)

# Calculate average demand for each season
seasonal_avg = data.groupby('Season')['Electric_demand'].mean()
print(seasonal_avg)

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(demand, label="Electricity Demand")
plt.title("Electricity Demand Time Series")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.show()

# Make the series stationary (if needed)
from statsmodels.tsa.stattools import adfuller

result = adfuller(demand)
if result[1] > 0.05:  # If p-value > 0.05, apply differencing
    demand_diff = demand.diff().dropna()
    demand = demand_diff

# Determine ARIMA (p, d, q) using PACF and ACF plots
plot_acf(demand)
plt.show()

plot_pacf(demand)
plt.show()
# Fit SARIMAX model
exog = pd.DataFrame({'Temperature': temperature, 'DayOfWeek': days})
model = SARIMAX(demand, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# Forecast
forecast_steps = 10  # Number of steps to forecast
forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog[-forecast_steps:])
forecast_mean = forecast.predicted_mean

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(demand, label="Observed")
plt.plot(forecast_mean, label="Forecast", color='red')
plt.title("Electricity Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.show()

# Calculate mean squared error
mse = mean_squared_error(demand[-forecast_steps:], forecast_mean)
print(f'Mean Squared Error: {mse}')