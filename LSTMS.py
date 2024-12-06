import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load time series data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)
time_series = data['Electric_demand']  # Replace 'Electric_demand' with your column name for electricity demand

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse scale the predictions and actual values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate confidence intervals (assuming normal distribution of errors)
train_errors = y_train_actual - train_predictions
test_errors = y_test_actual - test_predictions

train_error_std = np.std(train_errors)
test_error_std = np.std(test_errors)

train_confidence_intervals = np.array([train_predictions - 1.96 * train_error_std, train_predictions + 1.96 * train_error_std]).T
test_confidence_intervals = np.array([test_predictions - 1.96 * test_error_std, test_predictions + 1.96 * test_error_std]).T

# Plot the results with confidence intervals
plt.figure(figsize=(10, 5))
plt.plot(data.index[:len(train_predictions)], train_predictions, label="Train Predictions", color="green")
plt.fill_between(data.index[:len(train_predictions)], train_confidence_intervals[:, 0], train_confidence_intervals[:, 1], color='green', alpha=0.3)
plt.plot(data.index[len(train_predictions):len(train_predictions) + len(test_predictions)], test_predictions, label="Test Predictions", color="orange")
plt.fill_between(data.index[len(train_predictions):len(train_predictions) + len(test_predictions)], test_confidence_intervals[:, 0], test_confidence_intervals[:, 1], color='orange', alpha=0.3)
plt.title("LSTM Model Forecasting with Confidence Intervals")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate mean squared error
mse_train = mean_squared_error(y_train_actual, train_predictions)
mse_test = mean_squared_error(y_test_actual, test_predictions)
print(f'Mean Squared Error (Train): {mse_train}')
print(f'Mean Squared Error (Test): {mse_test}')

# Outlier analysis on temperature
temperature = data['Temperature']  # Replace 'Temperature' with your column name for temperature
temperature_mean = temperature.mean()
temperature_std = temperature.std()
outliers = temperature[(temperature < temperature_mean - 3 * temperature_std) | (temperature > temperature_mean + 3 * temperature_std)]

# Examine if anomalies coincide with significant deviations in forecast accuracy
forecast_errors = y_test_actual - test_predictions
significant_deviations = forecast_errors[(forecast_errors > forecast_errors.mean() + 3 * forecast_errors.std()) | (forecast_errors < forecast_errors.mean() - 3 * forecast_errors.std())]

# Print outliers and significant deviations
print("Temperature Outliers:")
print(outliers)
print("Significant Forecast Deviations:")
print(significant_deviations)