import subprocess
import sys
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from dotenv import load_dotenv
import openpyxl


# Function to install openpyxl
def install_openpyxl():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])


# Call the function to install openpyxl
install_openpyxl()


# Load the dataset
def load_data(filename):
    data = pd.read_csv(filename, parse_dates=["Time"])
    data = data.set_index("Time")
    data = data.dropna()
    return data


# Plotting functions
def plot_predictions(train, predictions, title, output_address):
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label="Actual")
    plt.plot(train.index, predictions, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Electric Demand")
    plt.legend()
    plt.savefig(os.path.join(output_address, title + ".jpg"))


def plot_errors(errors, title, output_address):
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="Prediction Errors")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(os.path.join(output_address, title + ".jpg"))


def plot_accuracy(mse, rmse, mae, title, output_address):
    metrics = ["MSE", "RMSE", "MAE"]
    values = [mse, rmse, mae]
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=["blue", "orange", "green"])
    plt.title(title)
    plt.savefig(os.path.join(output_address, title + ".jpg"))


# Data partitioning
def data_allocation(data, train_ratio=0.8):
    train_len = int(len(data) * train_ratio)
    train = data.iloc[:train_len]
    test = data.iloc[train_len:]
    return train, test


# Transform data for LSTM
def apply_transform(data, n):
    X, y = [], []
    for i in range(n, len(data)):
        X.append(data[i - n : i])
        y.append(data[i])
    return np.array(X), np.array(y)


# LSTM model
def train_lstm(train, n, number_nodes, learning_rate, epochs, batch_size):
    X_train, y_train = apply_transform(train, n)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n, 1)),
            tf.keras.layers.LSTM(number_nodes),
            tf.keras.layers.Dense(number_nodes, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"]
    )
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1
    )  # Set verbose to 1 for detailed output

    return model


# Calculate accuracy
def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae


# ARIMA model
def train_arima(errors, order):
    model = ARIMA(errors, order=order)
    model_fit = model.fit()
    return model_fit


# Save the DataFrame to an Excel file and adjust column width
def save_to_excel(df, output_address, filename):
    excel_path = os.path.join(output_address, filename)
    df.to_excel(excel_path, index=False)

    # Load the workbook and select the active worksheet
    workbook = openpyxl.load_workbook(excel_path)
    worksheet = workbook.active

    # Set the width of the columns
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # Get the column name
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        adjusted_width = max_length + 2
        worksheet.column_dimensions[column_letter].width = adjusted_width

    # Save the workbook
    workbook.save(excel_path)


# Main function
def main():
    # Load environment variables
    load_dotenv()
    filename = os.getenv("FILE_ADDRESS")
    output_address = os.getenv("OUTPUT_ADDRESS")
    epochs = os.getenv("EPOCHS")
    learning_rate = os.getenv("LEARNING_RATE")
    batch_size = os.getenv("BATCH_SIZE")
    number_nodes = os.getenv("NUMBER_NODES")
    days = os.getenv("Prediction_days")
    n = os.getenv("NN_LAGS")

    # Check if environment variables are loaded correctly
    if None in [
        filename,
        output_address,
        epochs,
        learning_rate,
        batch_size,
        number_nodes,
        days,
        n,
    ]:
        raise ValueError(
            "One or more environment variables are not set. Please check your .env file."
        )

    # Convert environment variables to appropriate types
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)
    number_nodes = int(number_nodes)
    days = int(days)
    n = int(n)

    # Load and preprocess data
    data = load_data(filename)
    train, test = data_allocation(data["Electric_demand"], train_ratio=0.8)

    # Train LSTM model
    lstm_model = train_lstm(
        train.values, n, number_nodes, learning_rate, epochs, batch_size
    )

    # Make predictions with LSTM
    X_test, y_test = apply_transform(test.values, n)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_predictions = lstm_model.predict(X_test).flatten()

    # Calculate errors
    errors = y_test - lstm_predictions

    # Train ARIMA model on errors with fixed parameters (5,1,5)
    arima_order = (5, 1, 5)
    arima_model = train_arima(errors, arima_order)
    arima_predictions = arima_model.predict(start=0, end=len(errors) - 1)

    # Combine LSTM and ARIMA predictions
    final_predictions = lstm_predictions + arima_predictions

    # Calculate accuracy
    mse, rmse, mae = calculate_accuracy(y_test, final_predictions)

    # Plot results
    plot_predictions(test[n:], final_predictions, "Final Predictions", output_address)
    plot_errors(errors, "Prediction Errors", output_address)
    plot_accuracy(mse, rmse, mae, "Model Accuracy", output_address)

    # Print results
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")

    # Reset index to access 'Time' column
    data = data.reset_index()

    # Create a DataFrame with the results
    results_df = data.copy()
    results_df["LSTM_Predicted"] = np.nan
    results_df["Final_Predicted"] = np.nan
    results_df.iloc[-len(y_test) :, results_df.columns.get_loc("LSTM_Predicted")] = (
        lstm_predictions
    )
    results_df.iloc[-len(y_test) :, results_df.columns.get_loc("Final_Predicted")] = (
        final_predictions
    )

    # Save the DataFrame to an Excel file and adjust column width
    save_to_excel(results_df, output_address, "predictions.xlsx")


if __name__ == "__main__":
    main()
