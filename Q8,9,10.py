import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller  # Import adfuller

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
# train_data, test_data, train_temperature, test_temperature

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test_data))

# Fit LSTM model
seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, batch_size=1, epochs=10)
lstm_forecast = lstm_model.predict(X_test)

# Calculate confidence intervals
def calculate_confidence_interval(predictions, actual, confidence=0.95):
    errors = predictions - actual
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * std_error
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

arima_lower, arima_upper = calculate_confidence_interval(arima_forecast, test_data)
lstm_lower, lstm_upper = calculate_confidence_interval(lstm_forecast, y_test)

# Compare intervals across different times (e.g., day, month)
# Assuming test_data has a datetime index
arima_intervals = pd.DataFrame({'lower': arima_lower, 'upper': arima_upper}, index=test_data.index)
lstm_intervals = pd.DataFrame({'lower': lstm_lower, 'upper': lstm_upper}, index=test_data.index)

# Group by day and month to check stability
arima_daily = arima_intervals.resample('D').mean()
arima_monthly = arima_intervals.resample('M').mean()
lstm_daily = lstm_intervals.resample('D').mean()
lstm_monthly = lstm_intervals.resample('M').mean()

print("ARIMA Daily Intervals:\n", arima_daily)
print("ARIMA Monthly Intervals:\n", arima_monthly)
print("LSTM Daily Intervals:\n", lstm_daily)
print("LSTM Monthly Intervals:\n", lstm_monthly)