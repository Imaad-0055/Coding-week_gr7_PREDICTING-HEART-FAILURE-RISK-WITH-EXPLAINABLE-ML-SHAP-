# Test SHAP Explanation Code

import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Test function to check SHAP explanation flow 
def test_shap_explanation():
    try:
        # Load test data
        X_train = pd.read_csv("X_train_resampled.csv")
        X_test = pd.read_csv("X_test.csv")

        # Drop the 'time' column
        X_train = X_train.drop("time", axis=1)
        X_test = X_test.drop("time", axis=1)

        # Drop the 'Unnamed: 0' column if it exists
        X_train = X_train.drop(columns=['Unnamed: 0'], errors='ignore')
        X_test = X_test.drop(columns=['Unnamed: 0'], errors='ignore')

        # Load models
        R_F = joblib.load('Random Forest.joblib')
        XGB = joblib.load('XGBoost.joblib')
        LGB = joblib.load('LightGBM.joblib')

        # Get feature names (assuming X_train is a pandas DataFrame)
        feature_names = X_train.columns.tolist()

        # 1. Test SHAP for XGBoost
        explainer_xgb = shap.TreeExplainer(XGB)
        shap_values_xgb = explainer_xgb(X_test)  # Get SHAP values for XGBoost

        assert shap_values_xgb is not None, "SHAP values for XGBoost are None"

        # 2. Test SHAP for Random Forest
        def rf_predict(data):
            if hasattr(R_F, "predict_proba"):
                return R_F.predict_proba(data)[:, 1]  # Probability of positive class
            return R_F.predict(data)  # For regression

        explainer_rf = shap.KernelExplainer(rf_predict, shap.sample(X_train, 100))  # Sample background data
        shap_values_rf = explainer_rf.shap_values(X_test)

        assert shap_values_rf is not None, "SHAP values for Random Forest are None"

        # 3. Test SHAP for LightGBM
        explainer_lgb = shap.TreeExplainer(LGB)
        shap_values_lgb = explainer_lgb(X_test)  # Get SHAP values for LightGBM

        assert shap_values_lgb is not None, "SHAP values for LightGBM are None"

        # If the assertions pass, display the SHAP plots
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("XGBoost SHAP Summary")
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_rf, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("Random Forest SHAP Summary")
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_lgb, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("LightGBM SHAP Summary")
        plt.show()

        print("SHAP explanation test passed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")

# Run the test
test_shap_explanation()
