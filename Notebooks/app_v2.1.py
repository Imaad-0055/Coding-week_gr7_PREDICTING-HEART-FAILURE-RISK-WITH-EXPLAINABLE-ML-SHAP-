import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import boxcox
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from data_processing import optimize_memory
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
from datetime import datetime


# Set page configuration
st.set_page_config(
    page_title="Heart Failure Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load models from files with error handling
try:
    # Create a models dictionary with only Random Forest
    rf = joblib.load("/Users/m1/Desktop/CODING_WEEK/Random Forest.joblib")
    
    # Set default model
    current_model = "Random Forest"
    
    # Define model information dictionary
    model_info = {
        "Random Forest": {
            "accuracy": "85%",
            "f1_score": "0.83",
            "description": "An ensemble learning method that builds multiple decision trees and merges their predictions.",
            "strengths": "Handles complex relationships well, less prone to overfitting."
        }
    }
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()# Load models from files with error handling


# styling using css
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Premium background with overlay gradient */
        .stApp {
            background: linear-gradient(135deg, rgba(10, 15, 30, 0.92), rgba(20, 30, 45, 0.95)),
                        url("app/static/bg.png") 
                        center/cover fixed;
            color: white;
            min-height: 100vh;
        }
        
        /* Title styling - centered with 3D effect */
        h1 {
            text-align: center;
            color: white;
            font-size: 42px;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(125, 211, 252, 0.7),
                         0 0 20px rgba(56, 189, 248, 0.5),
                         0 0 30px rgba(2, 132, 199, 0.3);
            margin-bottom: 30px;
            letter-spacing: 1px;
        }
        
        /* Subtitle styling */
        h2 {
            color: #7dd3fc;
            text-align: center;
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 20px;
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
            color: #7dd3fc; 
            border: 1px solid #7dd3fc;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 500;
            transition: all 0.3s ease;
            margin: 0 auto;
            display: block;
        }
        
        /* Button hover effect */
        .stButton > button:hover {
            background-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 0 12px rgba(125, 211, 252, 0.6);
            transform: translateY(-2px);
        }
        
        /* Custom input styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.7);
            color: white;
            border: 1px solid rgba(125, 211, 252, 0.3);
            border-radius: 6px;
            padding: 10px;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #7dd3fc;
            box-shadow: 0 0 8px rgba(125, 211, 252, 0.6);
            background-color: rgba(15, 23, 42, 0.85);
        }
        
        /* Label text styling */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label {
            color: #94e2ff;
            font-weight: 500;
            font-size: 16px;
            margin-bottom: 5px;
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div > div {
            background-color: rgba(15, 23, 42, 0.7);
            color: #7dd3fc !important;
            border: 1px solid rgba(125, 211, 252, 0.3);
            border-radius: 6px;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background-color: rgba(15, 23, 42, 0.6);
            border-radius: 6px;
            padding: 8px;
            border: 1px solid rgba(125, 211, 252, 0.2);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.2);
            border-left: 4px solid #22c55e;
            padding: 15px;
            border-radius: 4px;
            backdrop-filter: blur(3px);
        }
        
        /* Error message styling */
        .stError {
            background-color: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 4px;
            backdrop-filter: blur(3px);
        }
        
        /* Warning message styling */
        .stWarning {
            background-color: rgba(234, 179, 8, 0.2);
            border-left: 4px solid #eab308;
            padding: 15px;
            border-radius: 4px;
            backdrop-filter: blur(3px);
        }
        
        /* Info message styling */
        .stInfo {
            background-color: rgba(59, 130, 246, 0.2);
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 4px;
            backdrop-filter: blur(3px);
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
            font-size: 14px;
        }
        
        /* Container styling */
        .css-1d391kg, .css-12oz5g7 {
            background-color: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(125, 211, 252, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Section headers */
        h3 {
            color: #7dd3fc;
            border-bottom: 1px solid rgba(125, 211, 252, 0.3);
            padding-bottom: 8px;
            margin-top: 20px;
            font-size: 20px;
            font-weight: 600;
        }
        
        /* Input section container */
        .input-section {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid rgba(125, 211, 252, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(8px);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.4);
            border-radius: 8px;
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 6px;
            color: white;
            padding: 10px 16px;
            border: 1px solid rgba(125, 211, 252, 0.2);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(59, 130, 246, 0.2);
            border-color: rgba(125, 211, 252, 0.5);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.3) !important;
            border-color: #7dd3fc !important;
        }
        
        /* Metric styling */
        .stMetric {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid rgba(125, 211, 252, 0.3);
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .stMetric:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            border-color: rgba(125, 211, 252, 0.6);
        }
        
        /* Override for metric value */
        .css-1xarl3l {
            color: #7dd3fc;
            font-size: 2rem !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8);
            border-right: 1px solid rgba(125, 211, 252, 0.2);
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        /* Sidebar header */
        [data-testid="stSidebar"] h2 {
            color: #7dd3fc;
            margin-bottom: 20px;
            text-align: left;
            font-size: 22px;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 8px;
            padding: 5px;
            border: 1px solid rgba(125, 211, 252, 0.3);
        }
        
        /* Patient risk badge - high risk */
        .high-risk-badge {
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
            border: 1px solid rgba(239, 68, 68, 0.5);
        }
        
        /* Patient risk badge - moderate risk */
        .moderate-risk-badge {
            background-color: rgba(234, 179, 8, 0.2);
            color: #eab308;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
            border: 1px solid rgba(234, 179, 8, 0.5);
        }
        
        /* Patient risk badge - low risk */
        .low-risk-badge {
            background-color: rgba(34, 197, 94, 0.2);
            color: #22c55e;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
            border: 1px solid rgba(34, 197, 94, 0.5);
        }
        
        /* Chart container */
        .chart-container {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 12px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid rgba(125, 211, 252, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Footer styling */
        footer {
            text-align: center;
            padding: 20px 0;
            color: #94a3b8;
            font-size: 14px;
            border-top: 1px solid rgba(125, 211, 252, 0.1);
            margin-top: 30px;
        }
        
        /* Compare card */
        .compare-card {
            background-color: rgba(15, 23, 42, 0.7);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(125, 211, 252, 0.2);
            transition: all 0.3s ease;
        }
        
        .compare-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            border-color: rgba(125, 211, 252, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
apply_custom_styles()

# Add custom CSS for better dropdown contrast
st.markdown("""
<style>
/* Improved dropdown styling for better contrast */
.stSelectbox > div > div > div {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: white !important;
    border: 1px solid rgba(125, 211, 252, 0.5) !important;
}

/* Dropdown options styling */
.stSelectbox > div > div > div > div > div {
    background-color: rgba(15, 23, 42, 0.95) !important;
    color: white !important;
}

/* Dropdown option hover */
.stSelectbox > div > div > div > div > div:hover {
    background-color: rgba(59, 130, 246, 0.3) !important;
}

/* Selected option styling */
.stSelectbox [data-baseweb="selected-option"] {
    color: white !important;
    background-color: rgba(15, 23, 42, 0.95) !important;
}

/* Arrow icon color */
.stSelectbox [data-baseweb="select"] svg {
    color: rgba(125, 211, 252, 0.8) !important;
}

/* Focus state */
.stSelectbox [data-baseweb="select"]:focus-within {
    border-color: rgba(125, 211, 252, 0.8) !important;
    box-shadow: 0 0 8px rgba(125, 211, 252, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# Simplify sidebar for single model
st.sidebar.markdown("<h3>🧠 Model Information</h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style='background-color: rgba(15, 23, 42, 0.7); padding: 10px; border-radius: 5px; border: 1px solid rgba(125, 211, 252, 0.2);'>
    <p><strong>Random Forest Model:</strong> {model_info["Random Forest"]['description']}</p>
    <p>📊 <strong>Metrics:</strong><br>
       • Accuracy: {model_info["Random Forest"]['accuracy']}<br>
       • F1 Score: {model_info["Random Forest"]['f1_score']}</p>
    <p>💪 <strong>Strengths:</strong> {model_info["Random Forest"]['strengths']}</p>
</div>
""", unsafe_allow_html=True)

# Add a visual indicator of the current model in use
st.sidebar.markdown(f"""
<div style='margin-top: 10px; padding: 8px; border-radius: 4px; background-color: rgba(59, 130, 246, 0.2); border-left: 4px solid #3b82f6;'>
    <p style='margin: 0;'><strong>Active Model:</strong> Random Forest</p>
</div>
""", unsafe_allow_html=True)

# Data path and loading with proper error handling
DATA_PATH = "/Users/m1/Desktop/CODING_WEEK/heart_failure_clinical_records_dataset.csv"
try:
    df = optimize_memory(pd.read_csv(DATA_PATH))
    st.sidebar.success(f"Dataset loaded: {len(df)} records")
except FileNotFoundError:
    st.sidebar.error(f"Dataset file not found at {DATA_PATH}. Please check the file path.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading dataset: {e}")
    st.stop()

# Features setup
cont_features = ["serum_sodium", "serum_creatinine", "ejection_fraction", "creatinine_phosphokinase"]
other_features = ["age", "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
features = cont_features + other_features + ["platelets"]
# Boxcox lambda values
boxcox_lambdas = {
    'serum_sodium': 1.00536,
    'serum_creatinine': 1.00622,
    'ejection_fraction': 0.97857,
    'creatinine_phosphokinase': 0.82613
}


# Create folder for patient history
PATIENT_HISTORY_DIR = "/Users/m1/Desktop/CODING_WEEK/patient_history"
os.makedirs(PATIENT_HISTORY_DIR, exist_ok=True)

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

# Function to create a risk gauge chart
def create_risk_gauge(risk_percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Mortality Risk", 'font': {'color': 'white', 'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(125, 211, 252, 0.8)"},
            'bgcolor': "rgba(0, 0, 0, 0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=300,
        margin=dict(l=30, r=30, b=30, t=50)
    )
    
    return fig

def load_patient_history(patient_id):
    """
    Load patient history from a JSON file
    
    Parameters:
    patient_id (str): The patient ID to load history for
    
    Returns:
    list: A list of patient records, or None if the file doesn't exist
    """
    try:
        patient_file = os.path.join(PATIENT_HISTORY_DIR, f"patient_{patient_id}.json")
        
        if os.path.exists(patient_file):
            with open(patient_file, "r") as f:
                history = json.load(f)
                return history
        else:
            return None
    except Exception as e:
        st.error(f"Error loading patient history: {str(e)}")
        return None
def save_patient_data(patient_id, input_data, prediction, probability):
    """
    Save patient data to a JSON file with proper error handling
    """
    try:
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure the directory exists
        os.makedirs(PATIENT_HISTORY_DIR, exist_ok=True)
        
        # Prepare data to save - convert numpy values to Python native types for JSON serialization
        patient_data = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "input_data": {k: float(v) if isinstance(v, np.number) else v 
                          for k, v in input_data.to_dict(orient="records")[0].items()},
            "prediction": int(prediction),
            "death_probability": float(probability[1]),
            "survival_probability": float(probability[0]),
            "model_used": current_model
        }
        
        # Define the patient file path
        patient_file = os.path.join(PATIENT_HISTORY_DIR, f"patient_{patient_id}.json")
        
        # Check if file exists and load existing data
        if os.path.exists(patient_file):
            with open(patient_file, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []
        
        # Add new record to history
        history.append(patient_data)
        
        # Save updated history
        with open(patient_file, "w") as f:
            json.dump(history, f, indent=4)
        
        return patient_file
        
    except Exception as e:
        st.error(f"Error saving patient data: {str(e)}")
        return None

def create_pdf_report(patient_id, patient_data, prediction_results, model_name):
    """
    Generate a PDF report for the patient with prediction results and recommendations
    
    Parameters:
    patient_id (str): Patient identifier
    patient_data (pd.DataFrame): DataFrame containing patient health data
    prediction_results (dict): Dictionary containing prediction results
    model_name (str): Name of the model used for prediction
    
    Returns:
    bytes: PDF file as bytes
    """
    # Extract prediction results
    prediction = prediction_results['prediction']
    death_probability = prediction_results['death_probability']
    survival_probability = prediction_results['survival_probability']
    
    # Create a buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor("#0369a1"),
        spaceAfter=12
    )
    
    subheader_style = ParagraphStyle(
        'SubHeaderStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=6
    )
    
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    
    # Add the title
    elements.append(Paragraph(f"Heart Failure Risk Assessment Report", header_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add patient information section
    elements.append(Paragraph(f"Patient Information", subheader_style))
    
    # Format patient ID and date
    patient_info = [
        ["Patient ID:", f"{patient_id}"],
        ["Report Date:", f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        ["Model Used:", f"{model_name}"]
    ]
    
    patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (1, 0), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
    ]))
    
    elements.append(patient_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add clinical measurements section
    elements.append(Paragraph("Clinical Measurements", subheader_style))
    
    # Extract patient data for the table
    measurements = []
    measurements.append(["Parameter", "Value", "Reference Range"])
    
    # Add patient measurements from patient_data DataFrame
    row_data = patient_data.iloc[0]
    
    # Define reference ranges
    ref_ranges = {
        "age": "Adult",
        "anaemia": "0=No, 1=Yes",
        "creatinine_phosphokinase": "10-120 mcg/L",
        "diabetes": "0=No, 1=Yes",
        "ejection_fraction": "55-70%",
        "high_blood_pressure": "0=No, 1=Yes",
        "platelets": "150,000-450,000/mL",
        "serum_creatinine": "0.6-1.2 mg/dL",
        "serum_sodium": "135-145 mEq/L",
        "sex": "0=Female, 1=Male",
        "smoking": "0=No, 1=Yes"
    }
    
    # Add readable names and format values
    measurement_names = {
        "age": "Age (years)",
        "anaemia": "Anaemia",
        "creatinine_phosphokinase": "Creatinine Phosphokinase (CPK)",
        "diabetes": "Diabetes",
        "ejection_fraction": "Ejection Fraction (%)",
        "high_blood_pressure": "High Blood Pressure",
        "platelets": "Platelets (kiloplatelets/mL)",
        "serum_creatinine": "Serum Creatinine (mg/dL)",
        "serum_sodium": "Serum Sodium (mEq/L)",
        "sex": "Sex",
        "smoking": "Smoking"
    }
    
    # Format values appropriately
    for col in patient_data.columns:
        value = row_data[col]
        
        # Format binary values
        if col in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
            formatted_value = "Yes" if value == 1 else "No"
        elif col == "sex":
            formatted_value = "Male" if value == 1 else "Female"
        elif col == "platelets":
            formatted_value = f"{value:,.1f}"
        elif col == "age":
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.1f}" if isinstance(value, float) else str(value)
            
        measurements.append([measurement_names.get(col, col), formatted_value, ref_ranges.get(col, "")])
    
    # Create the table
    clinical_table = Table(measurements, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        # Highlight abnormal values
        ('TEXTCOLOR', (1, 4), (1, 4), colors.red if row_data["ejection_fraction"] < 40 else colors.black),
        ('TEXTCOLOR', (1, 7), (1, 7), colors.red if row_data["serum_creatinine"] > 1.2 else colors.black),
        ('TEXTCOLOR', (1, 8), (1, 8), colors.red if row_data["serum_sodium"] < 135 or row_data["serum_sodium"] > 145 else colors.black),
    ]))
    
    elements.append(clinical_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add prediction results section
    elements.append(Paragraph("Prediction Results", subheader_style))
    
    # Create risk level text and color
    if death_probability > 70:
        risk_level = "HIGH RISK"
        risk_color = colors.red
        recommendation = "Immediate medical attention recommended"
    elif death_probability > 30:
        risk_level = "MODERATE RISK"
        risk_color = colors.orange
        recommendation = "Regular monitoring recommended"
    else:
        risk_level = "LOW RISK"
        risk_color = colors.green
        recommendation = "Routine follow-up advised"
    
    # Create prediction results table
    prediction_data = [
        ["Mortality Risk:", f"{death_probability:.1f}%"],
        ["Survival Probability:", f"{survival_probability:.1f}%"],
        ["Risk Level:", risk_level],
        ["Recommendation:", recommendation]
    ]
    
    prediction_table = Table(prediction_data, colWidths=[2*inch, 3*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (1, 0), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),  # Color the risk percentage
        ('TEXTCOLOR', (1, 2), (1, 2), risk_color),  # Color the risk level
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),  # Bold the risk level
    ]))
    
    elements.append(prediction_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add clinical recommendations section
    elements.append(Paragraph("Clinical Recommendations", subheader_style))
    
    # Generate recommendations based on patient data
    recommendations = []
    
    # General recommendation based on risk level
    if death_probability > 70:
        recommendations.append("Consider hospitalization and intensive monitoring.")
    elif death_probability > 30:
        recommendations.append("Schedule follow-up within 2-4 weeks.")
    else:
        recommendations.append("Schedule routine follow-up in 3 months.")
    
    # Specific recommendations based on clinical values
    if row_data["ejection_fraction"] < 30:
        recommendations.append("Critical ejection fraction detected. Consider echocardiography and cardiology consultation.")
    elif row_data["ejection_fraction"] < 40:
        recommendations.append("Reduced ejection fraction. Evaluate for heart failure with reduced ejection fraction (HFrEF).")
        
    if row_data["serum_creatinine"] > 2:
        recommendations.append("Elevated serum creatinine suggests renal dysfunction. Consider nephrology consultation.")
    elif row_data["serum_creatinine"] > 1.2:
        recommendations.append("Mild elevation in serum creatinine. Monitor renal function.")
        
    if row_data["serum_sodium"] < 135:
        recommendations.append("Hyponatremia detected. Evaluate fluid status and consider sodium restriction.")
        
    if row_data["high_blood_pressure"] == 1:
        recommendations.append("Hypertension may be contributing to cardiac stress. Evaluate blood pressure management.")
        
    if row_data["diabetes"] == 1:
        recommendations.append("Diabetes present - ensure glycemic control is optimized.")
        
    if row_data["smoking"] == 1:
        recommendations.append("Active smoking - recommend smoking cessation program.")
        
    if row_data["anaemia"] == 1:
        recommendations.append("Anemia present - consider further evaluation and treatment.")
    
    # Add recommendations as bullet points
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        bulletIndent=10,
        spaceBefore=0,
        bulletFontName='Symbol'
    )
    
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", bullet_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Add footer with disclaimer
    elements.append(Paragraph("Disclaimer: This report is generated by an AI system and should be reviewed by a healthcare professional before making clinical decisions. The prediction is based on statistical models and may not account for all patient-specific factors.", info_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the value of the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    return pdf_value

# Function to create a risk comparison chart
def create_risk_comparison_chart(feature_name, current_value, value_range, input_data, df):
    comparison_values = np.linspace(value_range[0], value_range[1], 10)
    risk_values = []
    
    for value in comparison_values:
        temp_data = input_data.copy()
        temp_data[feature_name] = [value]
        
        trans_data = transform_input_data(temp_data, df, cont_features, other_features, boxcox_lambdas)
        prob = rf.predict_proba(trans_data)[0][1] * 100
        risk_values.append(prob)
    
    fig = px.line(
        x=comparison_values, 
        y=risk_values,
        markers=True,
        labels={"x": feature_name, "y": "Mortality Risk (%)"},
        title=f"Impact of {feature_name} on Mortality Risk"
    )
    
    # Add a vertical line for current value
    fig.add_vline(
        x=current_value, 
        line_dash="dash", 
        line_color="rgba(125, 211, 252, 0.8)",
        annotation_text="Current Value",
        annotation_position="top right"
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font={'color': "white", 'family': "Arial"},
        title_font_color="white",
        legend_font_color="white",
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
        height=350,
        margin=dict(l=30, r=30, b=30, t=50)
    )
    
    return fig

# Main application header
st.markdown("<h1>❤️ Heart Failure Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h2>Advanced Clinical Decision Support System</h2>", unsafe_allow_html=True)

# Initialize session state
if 'patient_id' not in st.session_state:
    st.session_state['patient_id'] = ""
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
if 'last_patient_data' not in st.session_state:
    st.session_state['last_patient_data'] = None
if 'comparison_feature' not in st.session_state:
    st.session_state['comparison_feature'] = "age"

# Create tabs for different sections
tabs = st.tabs(["📊 Risk Assessment", "📈 Trend Analysis", "📋 Patient History"])

# Risk Assessment Tab
with tabs[0]:
    # Patient ID input for tracking
    col1, col2 = st.columns([2, 1])
    with col1:
        patient_id = st.text_input("Patient ID", value=st.session_state['patient_id'], 
                                   help="Enter a unique patient identifier for tracking")
    with col2:
        # Load patient button
        if st.button("📂 Load Patient Data"):
            if patient_id:
                patient_history = load_patient_history(patient_id)
                if patient_history:
                    st.session_state['patient_id'] = patient_id
                    st.session_state['last_patient_data'] = patient_history[-1]
                    st.success(f"Loaded data for Patient #{patient_id}")
                else:
                    st.warning(f"No history found for Patient #{patient_id}")
            else:
                st.warning("Please enter a Patient ID")
    
    # Create stylish sections for inputs
    st.markdown("<h3>Patient Health Data</h3>", unsafe_allow_html=True)
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    
    # Pre-fill values if patient data is loaded
    default_values = {}
    if st.session_state['last_patient_data']:
        default_values = st.session_state['last_patient_data']['input_data']
    
    # Create 3 columns for better layout of inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, 
                              value=default_values.get("age", 60), step=1)
        cpk = st.number_input("Creatinine Phosphokinase", min_value=0, max_value=10000, 
                              value=default_values.get("creatinine_phosphokinase", 100), step=10)
        ejection = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, 
                                  value=default_values.get("ejection_fraction", 38), step=1)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, max_value=1000000.0, 
                                   value=default_values.get("platelets", 262500.0), step=1000.0)
    
    with col2:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, 
                                          value=default_values.get("serum_creatinine", 1.1), step=0.1)
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=175, 
                                       value=default_values.get("serum_sodium", 136), step=1)
        # Use radio buttons instead of selectbox for better visibility
        st.write("Anaemia")
        anaemia = st.radio("Anaemia", ["No", "Yes"], horizontal=True, 
                          index=1 if default_values.get("anaemia", 0) == 1 else 0,
                          label_visibility="collapsed")
        anaemia = 1 if anaemia == "Yes" else 0
        
        st.write("Diabetes")
        diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True, 
                           index=1 if default_values.get("diabetes", 0) == 1 else 0,
                           label_visibility="collapsed")
        diabetes = 1 if diabetes == "Yes" else 0
    
    with col3:
        st.write("High Blood Pressure")
        high_bp = st.radio("High Blood Pressure", ["No", "Yes"], horizontal=True, 
                           index=1 if default_values.get("high_blood_pressure", 0) == 1 else 0,
                           label_visibility="collapsed")
        high_bp = 1 if high_bp == "Yes" else 0
        
        st.write("Sex")
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True, 
                       index=1 if default_values.get("sex", 0) == 1 else 0,
                       label_visibility="collapsed")
        sex = 1 if sex == "Male" else 0
        
        st.write("Smoking")
        smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True, 
                          index=1 if default_values.get("smoking", 0) == 1 else 0,
                          label_visibility="collapsed")
        smoking = 1 if smoking == "Yes" else 0
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Center the buttons using columns and add more options
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        predict_clicked = st.button("🔍 Predict Risk")
    
    with col2:
        reset_clicked = st.button("🔄 Reset Fields")
    
    with col3:
        save_clicked = st.button("💾 Save Patient Data")
        
    with col4:
        generate_report = st.button("📄 Generate Report")
    # Helper function to get current input data
def get_current_input_data():
    return pd.DataFrame({
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
     # Handle reset action
if reset_clicked:
    st.rerun()

if predict_clicked:
    if not patient_id:
        st.warning("Please enter a Patient ID before prediction")
    else:
        st.session_state['patient_id'] = patient_id
        input_data = get_current_input_data()
        # Transform and predict
        trans_data = transform_input_data(input_data, df, cont_features, other_features, boxcox_lambdas)
        
        # Use the current model (fix the model selection)
        model = rf  # Use Random Forest as it's the only model available
        
        with st.spinner('Analyzing patient data...'):
            try:
                # Get both the prediction and probability
                prediction = rf.predict(trans_data)[0]
                probability = rf.predict_proba(trans_data)[0]
                
                # Format the probability
                death_probability = probability[1] * 100  # Convert to percentage
                survival_probability = probability[0] * 100  # Convert to percentage
                predictions_results = {
                    'prediction': prediction,
                    'death_probability': death_probability,
                    'survival_probability': survival_probability
                }
                # Save the results in session state
                st.session_state['last_prediction'] = {
                    'prediction': prediction,
                    'death_probability': death_probability,
                    'survival_probability': survival_probability,
                    'input_data': input_data
                }
                st.session_state['prediction_made'] = True
                
                # Also save the current model name in session state if not already there
                if 'current_model' not in st.session_state:
                    st.session_state['current_model'] = "Random Forest"
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Model type:", type(rf))
                st.stop()  # Stop execution if prediction fails
        
        # Display prediction results
        st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
        
        # Display risk gauge
        risk_gauge = create_risk_gauge(death_probability)
        st.plotly_chart(risk_gauge, use_container_width=True)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mortality Risk", f"{death_probability:.1f}%")
        
        with col2:
            st.metric("Survival Probability", f"{survival_probability:.1f}%")
        
        with col3:
            risk_level = "Critical" if death_probability > 70 else "Moderate" if death_probability > 30 else "Stable"
            st.metric("Patient Status", risk_level)
        
        # Risk badge with classification
        if death_probability > 70:
            st.markdown("<div class='high-risk-badge'>HIGH RISK: Immediate medical attention recommended</div>", unsafe_allow_html=True)
        elif death_probability > 30:
            st.markdown("<div class='moderate-risk-badge'>MODERATE RISK: Regular monitoring recommended</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='low-risk-badge'>LOW RISK: Routine follow-up advised</div>", unsafe_allow_html=True)
        
        # Show detailed message with probability
        if prediction == 1:
            st.error(f"⚠️ The patient's condition is critical with a {death_probability:.1f}% probability of mortality. Immediate medical intervention is advised.")
            
            # Add specific recommendations based on input values
            recommendations = []
            if ejection < 30:
                recommendations.append("Critical ejection fraction detected. Consider echocardiography and cardiology consultation.")
            if serum_creatinine > 2:
                recommendations.append("Elevated serum creatinine suggests renal dysfunction. Consider nephrology consultation.")
            if high_bp == 1:
                recommendations.append("Hypertension may be contributing to cardiac stress. Evaluate blood pressure management.")
            
            if recommendations:
                st.warning("### Clinical Recommendations\n" + "\n".join([f"- {r}" for r in recommendations]))
        else:
            st.success(f"✅ The patient's condition is stable with a {survival_probability:.1f}% probability of survival. Regular monitoring is recommended.")
            
            # Add preventive recommendations
            if any([diabetes, smoking, high_bp]):
                st.info("### Preventive Recommendations")
                if diabetes:
                    st.markdown("- Monitor blood glucose levels regularly")
                if smoking:
                    st.markdown("- Implement smoking cessation program")
                if high_bp:
                    st.markdown("- Continue blood pressure management")
        
        # Feature importance visualization
        st.markdown("<h3>Risk Factor Analysis</h3>", unsafe_allow_html=True)
        
        # Let user select which feature to analyze
        feature_to_compare = st.selectbox(
            "Select feature to analyze impact on risk:",
            ["age", "ejection_fraction", "serum_creatinine", "serum_sodium", "platelets", "creatinine_phosphokinase"],
            index=0
        )
        
        # Define value ranges for each feature
        feature_ranges = {
            "age": [30, 95],
            "ejection_fraction": [10, 80],
            "serum_creatinine": [0.5, 9.0],
            "serum_sodium": [113, 150],
            "platelets": [25000, 850000],
            "creatinine_phosphokinase": [20, 5000]
        }
        
        # Create and display risk comparison chart
        comparison_chart = create_risk_comparison_chart(
            feature_to_compare, 
            input_data[feature_to_compare].values[0], 
            feature_ranges[feature_to_compare],
            input_data,
            df 
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Show interpretation
        current_value = input_data[feature_to_compare].values[0]
        if feature_to_compare == "ejection_fraction":
            if current_value < 30:
                st.markdown("**Interpretation:** Ejection fraction below 30% indicates severe heart failure.")
            elif current_value < 40:
                st.markdown("**Interpretation:** Ejection fraction between 30-40% indicates moderate heart failure.")
            else:
                st.markdown("**Interpretation:** Ejection fraction above 40% suggests preserved heart function.")
        elif feature_to_compare == "serum_creatinine":
            if current_value > 1.5:
                st.markdown("**Interpretation:** Elevated serum creatinine suggests impaired kidney function, which can worsen heart failure.")
        
       # Feature importance image with better layout
        try:
             image_path = "/Users/m1/Desktop/CODING_WEEK/Feature_importance.png"
             st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
     # Use st.image directly with the file path instead of PIL
             st.image(image_path, caption="Impact of Individual Features on Mortality Risk Prediction", use_container_width=True)
        except Exception as e:
             st.warning(f"Feature importance visualization not available: {str(e)}")
        
        # Show the input data for reference in an expandable section
        with st.expander("View Input Data Summary"):
            st.dataframe(input_data)
            
        # Generate PDF report
        if generate_report:
            st.info("📄 Generating clinical report... This feature will export a PDF summary with all patient data and recommendations.")
        # Generate the PDF report using the existing function
        with st.spinner("Generating report..."):
             try:
            # Create the PDF using the existing function
                  pdf_content = create_pdf_report(patient_id, input_data, predictions_results, "Random Forest")           
            # Get the absolute path for the directory
                  current_dir = os.path.abspath(os.getcwd())
                  report_dir = os.path.join(current_dir, "Patients Reports")
            # Create the directory
                  os.makedirs(report_dir, exist_ok=True)
            # Check if directory was created successfully
                  if not os.path.exists(report_dir):
                      st.error(f"Failed to create directory at {report_dir}")
                  else:
            # Absolute path for the file
                     file_path = os.path.join(report_dir, f"patient_{patient_id}_report.pdf")
                # Save the PDF to the directory
                     with open(file_path, "wb") as f:
                         f.write(pdf_content)
                # Verify file was written
                     if os.path.exists(file_path):
                         file_size = os.path.getsize(file_path)
                         st.success(f"Report successfully saved to {file_path} (Size: {file_size} bytes)")
                     else:
                         st.error(f"Failed to write file at {file_path}")
            
            # Also provide a download button for immediate download
                  st.download_button(
                label="Download Clinical Report",
                data=pdf_content,
                file_name=f"patient_{patient_id}_report.pdf",
                mime="application/pdf"
                  )
            
             except Exception as e:
                 st.error(f"Error generating report: {str(e)}")
                 import traceback
                 st.code(traceback.format_exc())
# Handle save button - independent from prediction
if save_clicked:
    if not patient_id:
        st.warning("Please enter a Patient ID before saving data")
    else:
        try:
            input_data = get_current_input_data()
            
            # Check if prediction was made - use those values if available
            if st.session_state.get('prediction_made', False):
                prediction = st.session_state['last_prediction']['prediction']
                probability = [
                    st.session_state['last_prediction']['survival_probability'] ,
                    st.session_state['last_prediction']['death_probability'] 
                ]
            else:
                # No prediction was made, use defaults
                prediction = 0  # Default to "stable" prediction
                probability = [1.0, 0.0]  # Default to 100% survival, 0% mortality
            
            # Save the data
            patient_file = save_patient_data(patient_id, input_data, prediction, probability)
            st.success(f"Patient data saved successfully to {patient_file}")
        except Exception as e:
            st.error(f"Error saving patient data: {e}")
# Trend Analysis Tab
with tabs[1]:
    st.markdown("<h3>Patient Trend Analysis</h3>", unsafe_allow_html=True)
    
    if patient_id:
        # Load patient history
        history = load_patient_history(patient_id)
        
        if history and len(history) > 1:
            # Extract data for plotting
            dates = [entry["timestamp"] for entry in history]
            risks = [entry["death_probability"] for entry in history]
            
            # Create trend chart
            fig = px.line(
                x=dates, 
                y=risks, 
                markers=True,
                labels={"x": "Date", "y": "Mortality Risk (%)"},
                title=f"Risk Trend for Patient #{patient_id}"
            )
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.3)",
                font={'color': "white", 'family': "Arial"},
                title_font_color="white",
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate risk change
            latest_risk = risks[-1]
            previous_risk = risks[-2]
            risk_change = latest_risk - previous_risk
            
            # Display risk change metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Risk", f"{latest_risk:.1f}%", f"{risk_change:.1f}%", 
                         delta_color="inverse")
            
            # Show key metrics changes
            if len(history) >= 2:
                st.markdown("<h3>Changes in Key Health Indicators</h3>", unsafe_allow_html=True)
                
                latest = history[-1]["input_data"]
                previous = history[-2]["input_data"]
                
                metrics_to_track = {
                    "ejection_fraction": "Ejection Fraction",
                    "serum_creatinine": "Serum Creatinine",
                    "serum_sodium": "Serum Sodium"
                }
                
                cols = st.columns(len(metrics_to_track))
                for i, (key, label) in enumerate(metrics_to_track.items()):
                    with cols[i]:
                        current_val = latest[key]
                        prev_val = previous[key]
                        change = current_val - prev_val
                        
                        # For some metrics (like creatinine), lower is better, so invert delta color
                        delta_color = "normal"
                        if key == "serum_creatinine":
                            delta_color = "inverse"
                        elif key == "ejection_fraction" or key == "serum_sodium":
                            # For these, higher is generally better
                            delta_color = "normal"
                            
                        st.metric(label, f"{current_val}", f"{change:.2f}", delta_color=delta_color)
                
                # Add interpretation
                st.markdown("<h4>Clinical Interpretation</h4>", unsafe_allow_html=True)
                interpretations = []
                
                ef_change = latest["ejection_fraction"] - previous["ejection_fraction"]
                if abs(ef_change) >= 5:
                    direction = "improved" if ef_change > 0 else "declined"
                    interpretations.append(f"Ejection fraction has {direction} by {abs(ef_change):.1f}%.")
                
                cr_change = latest["serum_creatinine"] - previous["serum_creatinine"]
                if abs(cr_change) >= 0.3:
                    direction = "improved" if cr_change < 0 else "worsened"
                    interpretations.append(f"Kidney function has {direction} (serum creatinine changed by {abs(cr_change):.2f} mg/dL).")
                
                # Display interpretations
                if interpretations:
                    for interp in interpretations:
                        st.markdown(f"- {interp}")
                else:
                    st.markdown("- No significant changes in key health indicators.")
        
        elif history and len(history) == 1:
            st.info("Only one record available for this patient. More data points are needed to show trends.")
        else:
            st.warning("No history found for this patient. Please save prediction data first.")
    else:
        st.info("Please enter a Patient ID and save prediction data to view trends.")

# Patient History Tab
with tabs[2]:
    st.markdown("<h3>Patient History Records</h3>", unsafe_allow_html=True)
    
    if patient_id:
        history = load_patient_history(patient_id)
        
        if history:
            # Display patient records in a more structured way
            for i, record in enumerate(history):
                with st.container():
                    st.markdown(f"<div class='compare-card'>", unsafe_allow_html=True)
                    
                    # Two columns - one for date and status, one for risk values
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Record {i+1}** - {record['timestamp']}")
                        st.markdown(f"**Status:** {'Critical' if record['prediction'] == 1 else 'Stable'}")
                        
                    with col2:
                        st.markdown(f"**Risk:** {record['death_probability']:.1f}%")
                        st.markdown(f"**Model:** {record['model_used']}")
                    
                    # Expandable section with full details
                    with st.expander("View Details"):
                        # Convert to DataFrame for better display
                        record_df = pd.DataFrame([record["input_data"]])
                        st.dataframe(record_df)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No history found for this patient. Please save prediction data first.")
    else:
        st.info("Please enter a Patient ID and save prediction data to view history.")

# Add explanatory features in the sidebar for doctors
with st.sidebar:
    st.markdown("### 📘 Clinical Guidelines")
    
    with st.expander("Heart Failure Risk Factors"):
        st.markdown("""
        - **Ejection Fraction:** < 40% indicates heart failure with reduced ejection fraction
        - **Serum Creatinine:** > 1.5 mg/dL suggests kidney dysfunction
        - **Serum Sodium:** < 135 mEq/L indicates hyponatremia, common in heart failure
        """)
    
    with st.expander("Interpretation Guide"):
        st.markdown("""
        - **High Risk (>70%):** Immediate medical attention, consider hospitalization
        - **Moderate Risk (30-70%):** Close monitoring, medication review
        - **Low Risk (<30%):** Standard follow-up, focus on preventive measures
        """)
    
    with st.expander("About This Model"):
        st.markdown(f"""
        - **Current Model:** {current_model}
        - **Based on:** Heart failure clinical data
        - **Features:** 11 clinical parameters
        - **Accuracy:** 85% (on validation set)
        """)

# Add footer
st.markdown("""
<footer>
    <p>Advanced Heart Failure Prediction System v2.0</p>
    <p>Developed by Team 7 | Last Updated: March 2025</p>
    <p>UI Design & Functionalities by Abdelouahed TAHRI </p>            
</footer>
""", unsafe_allow_html=True)

