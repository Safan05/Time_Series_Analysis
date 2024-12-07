import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Load dataset
data = pd.read_csv("Database.csv")  # Replace with your dataset file

# Variability: Variance, Standard Deviation, Range
print("\nMeasures of Variability:")
print("Variance:\n", data.var())
print("Standard Deviation:\n", data.std())
print("Range:\n", data.max() - data.min())

# Distribution plots for each numerical column
numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# Scatter plots and pair plots
print("\nScatter Plots and Pair Plots:")
sns.pairplot(data)
plt.show()

# Correlations
print("\nCorrelation Analysis:")
correlation_matrix = data.corr()
print(correlation_matrix)

# Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()