# Reducing Multilinearity (correlation)

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

df.describe()

# Heatmap to visualize correlation between features

plt.figure(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,cmap='crest')

plt.show()