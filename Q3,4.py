import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)

# Extract relevant columns
electric_demand = data['Electric_demand']
temperature = data['Temperature']
humidity = data['Humidity']
wind_speed = data['Wind_speed']

# Create a 'Season' column
def get_season(date):
    if date.month in [12, 1, 2]:
        return 'Winter'
    elif date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data['Season'] = data.index.map(get_season)

# Regression analysis with temperature only
X_temp = temperature.values.reshape(-1, 1)
y = electric_demand.values
model_temp = LinearRegression().fit(X_temp, y)
pred_temp = model_temp.predict(X_temp)
mae_temp = mean_absolute_error(y, pred_temp)
rmse_temp = np.sqrt(mean_squared_error(y, pred_temp))

# Regression analysis with additional weather variables
X_weather = data[['Temperature', 'Humidity', 'Wind_speed']]
model_weather = LinearRegression().fit(X_weather, y)
pred_weather = model_weather.predict(X_weather)
mae_weather = mean_absolute_error(y, pred_weather)
rmse_weather = np.sqrt(mean_squared_error(y, pred_weather))

print(f'MAE with temperature only: {mae_temp}')
print(f'RMSE with temperature only: {rmse_temp}')
print(f'MAE with additional weather variables: {mae_weather}')
print(f'RMSE with additional weather variables: {rmse_weather}')

# Plot actual vs. predicted values for both models
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y, pred_temp, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs. Predicted (Temperature Only)')
plt.xlabel('Actual Electric Demand')
plt.ylabel('Predicted Electric Demand')

plt.subplot(1, 2, 2)
plt.scatter(y, pred_weather, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs. Predicted (With Weather Variables)')
plt.xlabel('Actual Electric Demand')
plt.ylabel('Predicted Electric Demand')
plt.tight_layout()
plt.show()

# Plot residuals for both models
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(pred_temp, y - pred_temp, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals (Temperature Only)')
plt.xlabel('Predicted Electric Demand')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
plt.scatter(pred_weather, y - pred_weather, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals (With Weather Variables)')
plt.xlabel('Predicted Electric Demand')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# Assuming you have preprocessed data for ARIMA and LSTM models
# and have the error distributions (e.g., MAE or RMSE) for each model

# Example error distributions (replace with actual values)
arima_errors = np.random.normal(loc=0, scale=1, size=100000)  # Replace with actual ARIMA errors
lstm_errors = np.random.normal(loc=0, scale=1, size=100000)  # Replace with actual LSTM errors

# Perform paired t-test
t_stat, p_value_ttest = ttest_rel(arima_errors, lstm_errors)
print(f'Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}')

# Perform Wilcoxon signed-rank test
w_stat, p_value_wilcoxon = wilcoxon(arima_errors, lstm_errors)
print(f'Wilcoxon signed-rank test: W-statistic = {w_stat}, p-value = {p_value_wilcoxon}')

# Plot error distributions for ARIMA and LSTM models
plt.figure(figsize=(10, 6))
sns.boxplot(data=[arima_errors, lstm_errors], palette='coolwarm')
plt.xticks([0, 1], ['ARIMA', 'LSTM'])
plt.title('Error Distributions (ARIMA vs. LSTM)')
plt.ylabel('Error')
plt.show()