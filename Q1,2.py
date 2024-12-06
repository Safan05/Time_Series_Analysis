import pandas as pd
import numpy as np
from scipy.stats import f_oneway, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('Database.csv', index_col='Time', parse_dates=True)

# Extract relevant columns
electric_demand = data['Electric_demand']  # Replace with your actual column name
temperature = data['Temperature']  # Replace with your actual column name

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

# Perform ANOVA to test for significant differences in electricity demand across seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
seasonal_data = [electric_demand[data['Season'] == season] for season in seasons]

anova_result = f_oneway(*seasonal_data)
print(f'ANOVA result: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}')

# Plot electricity demand across seasons
plt.figure(figsize=(10, 6))
sns.boxplot(x='Season', y='Electric_demand', data=data, order=seasons)
plt.title('Electricity Demand Across Seasons')
plt.xlabel('Season')
plt.ylabel('Electricity Demand')
plt.show()

# Calculate Pearson correlation between temperature and electricity demand
pearson_corr, pearson_pvalue = pearsonr(temperature, electric_demand)
print(f'Pearson correlation: r = {pearson_corr}, p-value = {pearson_pvalue}')

# Calculate Spearman correlation between temperature and electricity demand
spearman_corr, spearman_pvalue = spearmanr(temperature, electric_demand)
print(f'Spearman correlation: rho = {spearman_corr}, p-value = {spearman_pvalue}')

# Plot the relationship between temperature and electricity demand
plt.figure(figsize=(10, 6))
sns.scatterplot(x=temperature, y=electric_demand)
plt.title('Temperature vs. Electricity Demand')
plt.xlabel('Temperature')
plt.ylabel('Electricity Demand')
plt.show()

# Plot the correlation heatmap
corr_matrix = data[['Temperature', 'Electric_demand']].corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()