# 📊 Customer Churn Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) 
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.25-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

🔍 Project Description

This project implements a machine learning–based customer churn prediction system using the Telco Customer Churn dataset.

The goal is to predict whether a customer is likely to leave (churn) or stay, based on demographic details, service usage, and billing information.

The complete workflow — from data loading to model training, evaluation, and model saving — is implemented in Python using Scikit-learn in a Jupyter Notebook environment.

📁 Dataset Used

Dataset Name: Telco Customer Churn

Source File: WA_Fn-UseC_-Telco-Customer-Churn.csv

Target Variable: Churn

Key Features Include:

Customer tenure

Monthly charges

Total charges

Contract type

Payment method

Internet and phone services

Gender and senior citizen status

⚠️ Note: CustomerID column is dropped during preprocessing as it is not relevant for prediction.

🧠 Workflow Implemented

1️⃣ Data Loading
-------------------
Loaded the dataset using Pandas and performed initial inspection:

import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.info()
df.describe()

2️⃣ Data Preprocessing
-------------------------
Converted TotalCharges to numeric

Handled missing values

Encoded categorical variables (Label Encoding)

Dropped irrelevant columns (customerID)

Split dataset into features (X) and target (y)

Train-test split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

3️⃣ Exploratory Data Analysis (EDA)
--------------------------------------
Visualized churn distribution and analyzed key features:

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Churn', data=df)
plt.show()


Analyzed impact of:

Contract type

Tenure

Monthly charges

Used Matplotlib and Seaborn for visualizations.

4️⃣ Model Building
---------------------
Trained and compared multiple machine learning models:

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Example with Random Forest:

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

5️⃣ Model Evaluation
-----------------------
Evaluated model using accuracy, confusion matrix, precision, recall, and F1-score:

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


✅ Result: Random Forest achieved the best performance.

6️⃣ Model Saving
-------------------
Saved the trained model using Pickle for reuse:

import pickle

pickle.dump(model, open("customer_churn_model.pkl", "wb"))


This allows reusing the model without retraining.

🛠️ Technologies Used

Language: Python

Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, Pickle

IDE: Jupyter Notebook

✅ Results
-----------------
Successfully predicted customer churn with good accuracy

Identified important churn-driving factors:

Contract type

Tenure

Monthly charges

Created a reusable trained ML model

🚀 Future Improvements

Hyperparameter tuning for higher accuracy

Deployment using Flask or FastAPI

Real-time churn prediction dashboard

Integration with live customer data

📌 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for solving a real-world business problem.

By predicting customer churn, organizations can take proactive steps to improve customer retention and reduce revenue loss.
