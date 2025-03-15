# python main

#Importing libraries and exploring the dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from dataloader import download_data
from data_processing import optimise_memory

# Reading data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.info()

df.describe()

# Heatmap to visualize correlation between features

plt.figure(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,cmap='crest')

plt.show()

d = pd.read_csv("heart_failure_clinical_records_dataset.csv")

def optimize_memory(df):
    """
    Optimize memory usage of a pandas DataFrame by downcasting numeric data types.

    Parameters:
    -----------
    df : pandas.DataFrame
    Returns:
    --------
    pandas.DataFrame
        A memory-optimized copy of the input DataFrame
    """
    # Create a copy of the dataframe to avoid modifying the original
    result = df.copy()

    # Memory usage before optimization
    start_memory = result.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before optimization: {start_memory:.2f} MB")

    # Optimize numeric columns
    for col in result.columns:
        col_type = result[col].dtype

        # Process numerical columns
        if pd.api.types.is_numeric_dtype(col_type):

            # Integers
            if pd.api.types.is_integer_dtype(col_type):
                # Get min and max values to determine the smallest possible type
                c_min = result[col].min()
                c_max = result[col].max()

                # Determine best integer type based on min and max values
                if c_min >= 0:  # For unsigned integers
                    if c_max < 255:
                        result[col] = result[col].astype(np.uint8)
                    elif c_max < 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        result[col] = result[col].astype(np.uint32)
                    else:
                        result[col] = result[col].astype(np.uint64)
                else:  # For signed integers
                    if c_min > -128 and c_max < 127:
                        result[col] = result[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767:
                        result[col] = result[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647:
                        result[col] = result[col].astype(np.int32)
                    else:
                        result[col] = result[col].astype(np.int64)

            # Floats
            elif pd.api.types.is_float_dtype(col_type):
                # Downcast to float32 if possible
                c_min = result[col].min()
                c_max = result[col].max()

                # Check if float32 range is sufficient
                # (approximate range: -3.4e38 to 3.4e38)
                if c_min > -3.4e38 and c_max < 3.4e38:
                    result[col] = result[col].astype(np.float32)
                else:
                    result[col] = result[col].astype(np.float64)

        # For object columns, convert to category if beneficial
        elif col_type == 'object':
            # Calculate the ratio of unique values to total values
            unique_ratio = result[col].nunique() / len(result)

            # If the ratio is small, it's beneficial to use categorical
            if unique_ratio < 0.5:  # This threshold can be adjusted
                result[col] = result[col].astype('category')

    # Memory usage after optimization
    end_memory = result.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage after optimization: {end_memory:.2f} MB")
    print(f"Memory reduced by: {100 * (start_memory - end_memory) / start_memory:.2f}%")

    return result
optimize_memory(d)

# Reading data
df = optimise_memory(pd.read_csv('heart_failure_clinical_records_dataset.csv'))

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
     

# The transformation panel for the input data (user interface)

# Reading data
df = optimise_memory(pd.read_csv('heart_failure_clinical_records_dataset.csv'))
     
def transform_input_data(input_data, data_before_trans, data_after_transf, cont_features, other_features, boxcox_lambdas):
    """
    Transforms new input data using precomputed transformations.

    - Applies Box-Cox transformation for continuous features using precomputed lambdas.
    - Applies Winsorization for 'platelets' based on original dataset percentiles.
    - Standardizes using precomputed mean and std from transformed data.

    Parameters:
    - input_data (pd.DataFrame): New input data to be transformed.
    - data_before_trans (pd.DataFrame): Dataset before transformations (used for Winsorization).
    - data_after_transf (pd.DataFrame): Dataset after transformations (used to extract mean & std for standardization).
    - cont_features (list): List of continuous features to apply Box-Cox.
    - other_features (list): List of features that should remain unchanged.
    - boxcox_lambdas (dict): Precomputed dictionary of Box-Cox lambda values for continuous features.

    Returns:
    - pd.DataFrame: Transformed and standardized input data.
    """
    data_transformed = input_data.copy()

    # Apply Box-Cox transformation to continuous features using precomputed lambdas
    for col in cont_features:
        if col in boxcox_lambdas and col in data_transformed.columns:
            lambda_val = boxcox_lambdas[col]
            data_transformed[col] = boxcox(data_transformed[col] + 1e-6, lambda_val)  # Avoid zero values

    # Apply Winsorization to 'platelets' using percentiles from data_before_trans
    if 'platelets' in data_transformed.columns and 'platelets' in data_before_trans.columns:
        lower_bound = np.percentile(data_before_trans['platelets'], 5)
        upper_bound = np.percentile(data_before_trans['platelets'], 95)
        data_transformed['platelets'] = np.clip(data_transformed['platelets'], lower_bound, upper_bound)

    # Ensure other features remain unchanged
    for col in other_features:
        if col in input_data.columns:
            data_transformed[col] = input_data[col]

    # Extract precomputed means and stds from data_after_transf
    stats = data_after_transf.describe().T
    means = stats["mean"]
    stds = stats["std"]

    # Standardize input data using precomputed mean & std
    for col in data_transformed.columns:
        if col != "DEATH_EVENT" and col in means and col in stds and stds[col] != 0:  # Avoid division by zero
            data_transformed[col] = (data_transformed[col] - means[col]) / stds[col]

    return data_transformed


# Imbalance handling using oversampling (Smote) & data separation after standarisation

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

# Loading the training dataset and training models
## Loading the training dataset

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier


X_train = pd.read_csv("X_train_resampled.csv")
X_train = X_train.drop("time",axis=1)
#Y_train = pd.read_csv("Y_train.csv")
X_test = pd.read_csv("X_test.csv")
X_test = X_test.drop("time",axis=1)
Y_test = pd.read_csv("Y_test.csv")
#X_train = pd.read_csv("X_train_resampled.csv")
Y_train = pd.read_csv("Y_train_resampled.csv")

## Training models

### Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
     
### XGBoost

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, Y_train)
import joblib
joblib.dump(xgb_model,"XGBoost.joblib")
from google.colab import files
files.download("XGBoost.joblib")

### LightGBM

# Initialisation du modèle
model_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Entraînement du modèle
model_lgb.fit(X_train, Y_train)
import joblib
joblib.dump(model_lgb,"LightGBM.joblib")
from google.colab import files
files.download("LightGBM.joblib")


# Model Evaluation

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": lgb.LGBMClassifier(),
}

results = {
    "Modèle": ["LightGBM", "Random Forest", "XGBoost"],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": [],
    "ROC-AUC": [],
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    results["Accuracy"].append(accuracy_score(Y_test, Y_pred))
    results["Precision"].append(precision_score(Y_test, Y_pred))
    results["Recall"].append(recall_score(Y_test, Y_pred))
    results["F1-score"].append(f1_score(Y_test, Y_pred))
    results["ROC-AUC"].append(roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))
    
    #print(f"{name} Performance:")
    #print("Accuracy:", accuracy_score(Y_test, Y_pred))
    #print("Precision:", precision_score(Y_test, Y_pred))
    #print("Recall:", recall_score(Y_test, Y_pred))
    #print("F1-score:", f1_score(Y_test, Y_pred))
    #print("ROC-AUC:", roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))
    #print("-" * 40)'''

# SHAP Explanation

"""SHAP Explanation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mEzUe5bExSWWfcfBkpsxBM95mP_XsUqt
"""

import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

X_train = pd.read_csv("X_train_resampled.csv")
X_test = pd.read_csv("X_test.csv")
X_train = X_train.drop("time",axis=1)
X_test = X_test.drop("time",axis=1)

# Drop the 'Unnamed: 0' column if it exists
X_train = X_train.drop(columns=['Unnamed: 0'], errors='ignore')
X_test = X_test.drop(columns=['Unnamed: 0'], errors='ignore')

# Load models and data
R_F = joblib.load('Random Forest.joblib')
XGB = joblib.load('XGBoost.joblib')
LGB = joblib.load('LightGBM.joblib')
# Get feature names (assuming X_train is a pandas DataFrame)
feature_names = X_train.columns.tolist()
# --------------------------------------------
# For XGBoost
# --------------------------------------------
explainer_xgb = shap.TreeExplainer(XGB)
shap_values_xgb = explainer_xgb(X_test) # or .shap_values(X_test)
# Handle multi-class outputs
if isinstance(shap_values_xgb, list):
 shap_values_xgb = shap_values_xgb[1] # Select class for explanation
# Plot with explicit feature names and figure sizing
plt.figure(figsize=(10, 6)) # Larger figure for visibility
shap.summary_plot(
 shap_values_xgb,
 X_test,
 feature_names=feature_names, # Critical for labels
 plot_type="bar",
 show=False
)
plt.title("XGBoost SHAP Summary", fontsize=14)
plt.yticks(fontsize=12) # Explicitly set y-axis font size
plt.tight_layout() # Prevent text cutoff
plt.show()
# --------------------------------------------
# For Random Forest
# --------------------------------------------
# Using KernelExplainer instead for Random Forest
# First, create a prediction function
def rf_predict(data):
    # For classification, return probabilities
    if hasattr(R_F, "predict_proba"):
        return R_F.predict_proba(data)[:, 1]  # Probability of positive class
    # For regression
    return R_F.predict(data)

# Create explainer
explainer_rf = shap.KernelExplainer(rf_predict, shap.sample(X_train, 100))  # Sample background data
shap_values_rf = explainer_rf.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_rf,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
# --------------------------------------------
# For LightGBM
# --------------------------------------------
explainer_lgb = shap.TreeExplainer(LGB)
shap_values_lgb = explainer_lgb(X_test) # or .shap_values(X_test)
if isinstance(shap_values_lgb, list):
 shap_values_lgb = shap_values_lgb[1] # Select class for explanation
plt.figure(figsize=(10, 6))
shap.summary_plot(
 shap_values_lgb,
 X_test,
 feature_names=feature_names, # Must re-specify
 plot_type="bar",
 show=False
)
plt.title("LightGBM SHAP Summary", fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# interface

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import boxcox
import joblib
from data_processing import optimize_memory

# Load models from files
rf = joblib.load("/Users/m1/Desktop/CODING_WEEK/Random Forest.joblib")

# styling using css
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Premium background with overlay gradient */
        .stApp {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.92)),
                        url("app/static/bg.png") 
                        center/cover fixed;
            color: white;
            min-height: 100vh;
        }
        
        /* Title styling - centered */
        h1 {
            text-align: center;
            color: white;
            font-size: 42px;
            font-weight: bold;
            text-shadow: 2px 2px 8px rgba(0, 162, 255, 0.6);
            margin-bottom: 30px;
            letter-spacing: 0.5px;
        }
        
        /* Center button container */
        .stButton {
            text-align: center;
            display: flex;
            justify-content: center;
            margin: 20px auto;
        }
        
        /* Button styling with moderate blue text */
        .stButton > button {
            background-color: rgba(30, 41, 59, 0.8);
            color: #7dd3fc; /* Moderate/light blue text */
            border: 1px solid #7dd3fc;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s ease;
            margin: 0 auto;
            display: block;
        }
        
        /* Button hover effect */
        .stButton > button:hover {
            background-color: rgba(59, 130, 246, 0.2);
            box-shadow: 0 0 10px rgba(125, 211, 252, 0.5);
            transform: translateY(-2px);
        }
        
        /* Custom input styling */
        .stTextInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.7);
            color: white;
            border: 1px solid rgba(125, 211, 252, 0.3);
            border-radius: 6px;
            padding: 10px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #7dd3fc;
            box-shadow: 0 0 5px rgba(125, 211, 252, 0.5);
        }
        
        /* Number input styling */
        .stNumberInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.7);
            color: white;
            border: 1px solid rgba(125, 211, 252, 0.3);
            border-radius: 6px;
            padding: 10px;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: #7dd3fc;
            box-shadow: 0 0 5px rgba(125, 211, 252, 0.5);
        }
        
        /* Label text styling */
        .stTextInput label, .stNumberInput label, .stSelectbox label {
            color: #7dd3fc;
            font-weight: 500;
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div > div {
            background-color: rgba(15, 23, 42, 0.7);
            color: #7dd3fc !important;
            border: 1px solid rgba(125, 211, 252, 0.3);
            border-radius: 6px;
        }
        
        /* Selectbox dropdown styling */
        .stSelectbox > div > div > div > div {
            color: #7dd3fc !important;
        }
        
        /* Selectbox options styling */
        div[data-baseweb="select"] > div > div > div > div > div > div {
            color: #7dd3fc !important;
        }
        
        /* Selectbox hover styling */
        div[data-baseweb="select"] > div > div:hover {
            border-color: #7dd3fc;
        }
        
        /* Selectbox focus styling */
        div[data-baseweb="select"] > div > div:focus {
            border-color: #7dd3fc;
            box-shadow: 0 0 5px rgba(125, 211, 252, 0.5);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.2);
            border-left: 4px solid #22c55e;
            padding: 15px;
            border-radius: 4px;
        }
        
        /* Error message styling */
        .stError {
            background-color: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 4px;
        }
        
        /* Text color and styling */
        p, div {
            color: white;
            line-height: 1.6;
        }
        
        /* Caption for images */
        .caption {
            font-style: italic;
            color: #94a3b8;
            text-align: center;
            padding-top: 8px;
        }
        
        /* Container styling */
        .css-1d391kg, .css-12oz5g7 {
            background-color: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(125, 211, 252, 0.2);
        }
        
        /* Section headers */
        h3 {
            color: #7dd3fc;
            border-bottom: 1px solid rgba(125, 211, 252, 0.3);
            padding-bottom: 8px;
            margin-top: 20px;
        }
        
        /* Input section container */
        .input-section {
            background-color: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(125, 211, 252, 0.2);
        }
        
        /* Center text elements */
        .st-bq {
            text-align: center;
        }
        
        /* Probability display */
        .probability-display {
            font-weight: bold;
            font-size: 1.1em;
            color: #7dd3fc;
        }
        
        /* Metric styling */
        .stMetric {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(125, 211, 252, 0.3);
        }
        
        /* Override for metric value */
        .css-1xarl3l {
            color: #7dd3fc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
apply_custom_styles()


# Center the title using columns
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.title("PREDICTION OF DEATH EVENT USING ADVANCED MACHINE LEARNING MODEL")

DATA_PATH = "/Users/m1/Desktop/CODING_WEEK/heart_failure_clinical_records_dataset.csv"
try:
    df = optimize_memory(pd.read_csv(DATA_PATH))
except FileNotFoundError:
    st.error(f"Dataset file not found at {DATA_PATH}. Please check the file path.")
    st.stop()

# Features setup
cont_features = ["serum_sodium", "serum_creatinine", "ejection_fraction", "creatinine_phosphokinase"]
other_features = ["age", "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
features = cont_features + other_features + ["platelets"]
boxcox_lambdas = {
    'serum_sodium': 1.00536,
    'serum_creatinine': 1.00622,
    'ejection_fraction': 0.97857,
    'creatinine_phosphokinase': 0.82613
}

def transform_input_data(input_data, df, cont_features, other_features, boxcox_lambdas):
    """
    Transforms new input data using precomputed transformations.
    - Box-Cox transformation for continuous features.
    - Winsorization for 'platelets' using dataset percentiles.
    - Standardization using precomputed mean and std.
    """
    data_transformed = optimize_memory(input_data.copy())

    # Apply Box-Cox transformation
    for col in cont_features:
        if col in boxcox_lambdas and col in data_transformed.columns:
            lambda_val = boxcox_lambdas[col]
            data_transformed[col] = boxcox(data_transformed[col] + 1e-6, lambda_val)

    # Apply Winsorization to 'platelets'
    if 'platelets' in data_transformed.columns and 'platelets' in df.columns:
        lower_bound = np.percentile(df['platelets'], 5)
        upper_bound = np.percentile(df['platelets'], 95)
        data_transformed['platelets'] = np.clip(data_transformed['platelets'], lower_bound, upper_bound)

    # Preserve other features
    for col in other_features:
        if col in input_data.columns:
            data_transformed[col] = input_data[col]

    # Standardization using dataset mean and std
    stats = df.describe().T
    means = stats["mean"]
    stds = stats["std"]

    for col in data_transformed.columns:
        if col in means and col in stds and stds[col] != 0:  # Avoid division by zero
            data_transformed[col] = (data_transformed[col] - means[col]) / stds[col]

    return data_transformed

# Create stylish sections for inputs
st.markdown("<h3>Patient Health Data</h3>", unsafe_allow_html=True)
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

# Create 3 columns for better layout of inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=60, step=1)
    cpk = st.number_input("Creatinine Phosphokinase", min_value=0, max_value=10000, value=100, step=10)
    ejection = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=38, step=1)
    platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, max_value=1000000.0, value=262500.0, step=1000.0)

with col2:
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=1.1, step=0.1)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=175, value=136, step=1)
    # Use radio buttons instead of selectbox for better visibility
    st.write("Anaemia")
    anaemia = st.radio("Anaemia", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    anaemia = 1 if anaemia == "Yes" else 0
    
    st.write("Diabetes")
    diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    diabetes = 1 if diabetes == "Yes" else 0

with col3:
    st.write("High Blood Pressure")
    high_bp = st.radio("High Blood Pressure", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    high_bp = 1 if high_bp == "Yes" else 0
    
    st.write("Sex")
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True, label_visibility="collapsed")
    sex = 1 if sex == "Male" else 0
    
    st.write("Smoking")
    smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    smoking = 1 if smoking == "Yes" else 0

st.markdown("</div>", unsafe_allow_html=True)

# Center the button using columns
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_clicked = st.button("Predict Mortality Risk")

if predict_clicked:
    # Create a dataframe from the inputs
    input_data = pd.DataFrame({
        "age": [age],
        "anaemia": [anaemia],
        "creatinine_phosphokinase": [cpk],
        "diabetes": [diabetes],
        "ejection_fraction": [ejection],
        "high_blood_pressure": [high_bp],
        "platelets": [platelets],
        "serum_creatinine": [serum_creatinine],
        "serum_sodium": [serum_sodium],
        "sex": [sex],
        "smoking": [smoking]
    })
    
    # Transform and predict
    trans_data = transform_input_data(input_data, df, cont_features, other_features, boxcox_lambdas)
    
    # Get both the prediction and probability
    prediction = rf.predict(trans_data)[0]
    probability = rf.predict_proba(trans_data)[0]
    
    # Format the probability
    death_probability = probability[1] * 100  # Convert to percentage
    survival_probability = probability[0] * 100  # Convert to percentage
    
    # Display prediction results
    st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mortality Risk", f"{death_probability:.1f}%")
    
    with col2:
        st.metric("Survival Probability", f"{survival_probability:.1f}%")
    
    with col3:
        status = "Critical" if prediction == 1 else "Stable"
        st.metric("Patient Status", status)
    
    # Show detailed message with probability
    if prediction == 1:
        st.error(f"⚠️ The patient's condition is critical with a {death_probability:.1f}% probability of mortality. Immediate medical intervention is advised.")
    else:
        st.success(f"✅ The patient's condition is stable with a {survival_probability:.1f}% probability of survival. Regular monitoring is recommended.")
    
    # Feature importance image
    image_path = "/Users/m1/Desktop/CODING_WEEK/Feature_importance.png"
    try:
        image = Image.open(image_path)
        st.image(image, caption="Impact of Individual Data Points on Mortality Risk", use_column_width=True)
    except FileNotFoundError:
        st.error("Feature importance image not found. Please check the file path.")

    # Show the input data for reference
    st.markdown("<h3>Input Summary</h3>", unsafe_allow_html=True)
    st.dataframe(input_data)