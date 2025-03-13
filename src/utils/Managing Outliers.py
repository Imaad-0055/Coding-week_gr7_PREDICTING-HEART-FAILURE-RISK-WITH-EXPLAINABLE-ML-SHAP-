# Managing Outliers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Reading data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.info()

## Retriving and visualizing outliers

# Function to detect outliers using IQR
def detect_outliers_iqr(df):
    outliers = {}
    for col in df.select_dtypes(include=np.number):  # Select solely numerical columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers

# Application on the dataframe
outliers_dict = detect_outliers_iqr(df)

# Displaying the outliers for each column
for col, values in outliers_dict.items():
    print(f"Outliers in {col}:")
    print(values, "\n")
    
# Loop through each numerical column and create a boxplot
for column in df.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
    
## Transforming outliers

df_copy=df.copy() # une copie de la data
df_copy.describe()

cont_features = ["serum_sodium","serum_creatinine","ejection_fraction","creatinine_phosphokinase"]
other_features = ["age","anaemia","diabetes","high_blood_pressure","sex","smoking","time"]
features = cont_features + other_features + ["platelets"]

# Dictionary to store Box-Cox lambdas
boxcox_lambdas = {}

#  Fit Box-Cox on the training data and store lambda values
for col in cont_features:
    if (df_copy[col] <= 0).any():  # Ensure all values are positive
        df_copy[col] += abs(df_copy[col].min()) + 1
    df_copy[col], lambda_val = boxcox(df_copy[col])  # Apply Box-Cox
    boxcox_lambdas[col] = lambda_val  # Store lambda value
    
# Loop through each numerical column and create a boxplot
for column in df_copy.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_copy[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
    
df_copy['platelets'] = winsorize(df_copy['platelets'], limits=[0.05, 0.05])  # Capping at 5th and 95th percentiles

plt.figure(figsize=(6, 4))
sns.boxplot(x=df_copy['platelets'])
plt.title('Boxplot of platelets ')
plt.show()
     
