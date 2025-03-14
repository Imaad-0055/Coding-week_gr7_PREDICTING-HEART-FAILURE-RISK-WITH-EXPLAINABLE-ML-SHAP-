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

