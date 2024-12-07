import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load time series data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)
time_series = data['Electric_demand']  # Replace 'Electric_demand' with your column name for electricity demand

# Filter data for the years 2019 and 2020
filtered_data = time_series['2019':'2020']

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_data.values.reshape(-1, 1))

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
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predict values for the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate error values
mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predictions)
mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Extend the data to include 2021 for prediction
extended_data = time_series['2019':'2021']
scaled_extended_data = scaler.transform(extended_data.values.reshape(-1, 1))

# Create sequences for 2021 prediction
X_extended, _ = create_sequences(scaled_extended_data, seq_length)

# Predict values for 2021
extended_predictions = model.predict(X_extended)
extended_predictions = scaler.inverse_transform(extended_predictions)

# Output the predicted values for 2021
predicted_2021 = extended_predictions[-len(time_series['2021']):]

# Create a DataFrame for the predicted values
predicted_2021_df = pd.DataFrame(predicted_2021, index=time_series['2021'].index, columns=['Predicted_Electric_Demand'])

# Save the predicted values to a CSV file
predicted_2021_df.to_csv('predicted_2021.csv')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(filtered_data.index[train_size + seq_length:], scaler.inverse_transform(test_data[seq_length:]), label='Actual')
plt.plot(filtered_data.index[train_size + seq_length:], test_predictions, label='Predicted')
plt.plot(time_series['2021'].index, predicted_2021, label='2021 Prediction', linestyle='--')
plt.title('Electricity Demand Prediction')
plt.xlabel('Date')
plt.ylabel('Electricity Demand')
plt.legend()
plt.show()