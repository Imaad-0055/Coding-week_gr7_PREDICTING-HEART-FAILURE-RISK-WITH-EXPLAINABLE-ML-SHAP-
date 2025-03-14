# Test de XGBoost classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset 
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.info()  

# Handle missing values (if any)
print(f"Missing values:\n{df.isnull().sum()}")

# Assuming 'DEATH_EVENT' is the target column and the rest are features
X = df.drop('DEATH_EVENT', axis=1)  # Features
y = df['DEATH_EVENT']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
model = XGBClassifier()

# Train the model
print("Training the XGBoost model...")
model.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For AUC-ROC

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)
clf_report = classification_report(y_test, y_pred, target_names=['No Event', 'Event'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Classification Report:\n{clf_report}")

# Confusion Matrix
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

