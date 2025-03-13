# Loading the training dataset and training models
## Loading the training dataset

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier


X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")
X_train = pd.read_csv("X_train_resampled.csv")
Y_train = pd.read_csv("Y_train_resampled.csv")

## Training models

### Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
     
### XGBoost

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, Y_train)

### LightGBM

# Initialisation du modèle
model_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Entraînement du modèle
model_lgb.fit(X_train, Y_train)