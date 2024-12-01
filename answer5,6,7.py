import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "predictionsForHybrid.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", parse_dates=["Time"])

# Filter data to start from the specified time
df = df[df["Time"] >= "2021-05-26T20:05"]

# Calculate model errors
df["Error"] = df["Electric_demand"] - df["Final_Predicted"]

# Define extreme temperature thresholds (example: top and bottom 10%)
extreme_temp_threshold_high = df["Temperature"].quantile(0.90)
extreme_temp_threshold_low = df["Temperature"].quantile(0.10)

# Filter extreme and average temperature days
extreme_temp_days = df[
    (df["Temperature"] >= extreme_temp_threshold_high)
    | (df["Temperature"] <= extreme_temp_threshold_low)
]
average_temp_days = df[
    (df["Temperature"] < extreme_temp_threshold_high)
    & (df["Temperature"] > extreme_temp_threshold_low)
]

# Conduct t-test or Mann-Whitney U test
t_stat, p_value = ttest_ind(
    extreme_temp_days["Error"], average_temp_days["Error"], equal_var=False
)
# Alternatively, use Mann-Whitney U test
# u_stat, p_value = mannwhitneyu(extreme_temp_days['Error'], average_temp_days['Error'])

print(f"T-test: t-statistic = {t_stat}, p-value = {p_value}")

# Cross-correlation analysis for lag effect
lags = range(1, 25)  # Example: 1 to 24 hours
correlations = [df["Electric_demand"].autocorr(lag) for lag in lags]

plt.plot(lags, correlations)
plt.xlabel("Lag (hours)")
plt.ylabel("Cross-correlation")
plt.title("Lag Effect of Temperature on Electricity Demand")
plt.show()

# Identify holidays and weekends
holidays_weekends = df[df["Day_of_the_week"].isin([0, 6])]
regular_days = df[~df["Day_of_the_week"].isin([0, 6])]

# Conduct t-test for holidays/weekends vs. regular weekdays
t_stat, p_value = ttest_ind(
    holidays_weekends["Error"], regular_days["Error"], equal_var=False
)

print(
    f"T-test for holidays/weekends vs. regular weekdays: t-statistic = {t_stat}, p-value = {p_value}"
)
