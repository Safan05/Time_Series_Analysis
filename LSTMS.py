import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load time series data
# Replace with your dataset (e.g., a CSV file or generated data)
data = pd.read_csv('your_time_series.csv', index_col='Date', parse_dates=True)
time_series = data['Value'].values  # Replace 'Value' with your column name

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.reshape(-1, 1))

# Prepare the data for the LSTM model
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 10  # Number of previous time steps to use for prediction
X, y = create_dataset(scaled_data, time_steps)

# Reshape X to fit the LSTM input format (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse scale the predictions and actual values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(data.index, time_series, label="Original Time Series", color="blue")
plt.plot(data.index[:len(train_predictions)], train_predictions, label="Train Predictions", color="green")
plt.plot(data.index[len(train_predictions):], test_predictions, label="Test Predictions", color="orange")
plt.title("LSTM Model Forecasting")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
