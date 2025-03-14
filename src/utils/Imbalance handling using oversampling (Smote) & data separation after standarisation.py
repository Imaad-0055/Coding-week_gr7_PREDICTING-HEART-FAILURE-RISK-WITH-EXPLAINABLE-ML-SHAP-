# Imbalance handling using oversampling (Smote) & data separation after standarisation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from data_processing import optimise_memory

# Reading data
df = optimise_memory(pd.read_csv('heart_failure_clinical_records_dataset.csv'))


cont_features = ["serum_sodium","serum_creatinine","ejection_fraction","creatinine_phosphokinase"]
other_features = ["age","anaemia","diabetes","high_blood_pressure","sex","smoking","time"]
features = cont_features + other_features + ["platelets"] 

df_copy = df.copy()
scaler = StandardScaler()
df_copy[features] = scaler.fit_transform(df_copy[features])
     
     
X = df_copy.drop(columns=['DEATH_EVENT'])
Y = df_copy['DEATH_EVENT']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

df_train = pd.DataFrame(X_train_resampled)
df_test = pd.DataFrame(X_test)

print("Training Set Statistics:")
print(df_train.describe())

print("\n Testing Set Statistics :")
print(df_test.describe())