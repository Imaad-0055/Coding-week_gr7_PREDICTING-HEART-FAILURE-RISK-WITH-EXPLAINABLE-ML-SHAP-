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
