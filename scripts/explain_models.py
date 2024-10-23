import os
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the datasets
fraud_data = pd.read_csv('../data/processed/fraud_data.csv')
credit_data = pd.read_csv('../data/processed/creditcard_data.csv')

# Fraud data target and features separation
X_fraud = fraud_data[['purchase_value', 'transaction_freq', 'transaction_velocity', 'hour_of_day', 'day_of_week', 'source_encoded', 'browser_encoded', 'sex_encoded']]
y_fraud = fraud_data['class']

# Credit card data target and features separation
X_credit = credit_data.drop('Class', axis=1)
y_credit = credit_data['Class']

# Load the trained models
with open('../models/fraud_detection_rf_model.pkl', 'rb') as model_file:
    rf_model_fraud = pickle.load(model_file)

with open('../models/creditcard_fraud_rf_model.pkl', 'rb') as model_file:
    rf_model_credit = pickle.load(model_file)

# Ensure the explainability directory exists
os.makedirs('../explainability', exist_ok=True)

# SHAP Explainability
def shap_explain(model, X, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary Plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f'../explainability/{model_name}_shap_summary_plot.png')
    plt.clf()

    # Force Plot (for the first instance)
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0], show=False, matplotlib=True)
    plt.savefig(f'../explainability/{model_name}_shap_force_plot.png')
    plt.clf()

    # Dependence Plot (for the first feature)
    shap.dependence_plot(0, shap_values[1], X, show=False)
    plt.savefig(f'../explainability/{model_name}_shap_dependence_plot.png')
    plt.clf()

# LIME Explainability
def lime_explain(model, X, model_name):
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['Non-Fraud', 'Fraud'], discretize_continuous=True)
    exp = explainer.explain_instance(X.iloc[0], model.predict_proba, num_features=10)

    # Feature Importance Plot
    exp.as_pyplot_figure()
    plt.savefig(f'../explainability/{model_name}_lime_feature_importance.png')
    plt.clf()

# Explain Fraud Detection Model
shap_explain(rf_model_fraud, X_fraud, 'fraud_detection_rf')
lime_explain(rf_model_fraud, X_fraud, 'fraud_detection_rf')

# Explain Credit Card Fraud Detection Model
shap_explain(rf_model_credit, X_credit, 'creditcard_fraud_rf')
lime_explain(rf_model_credit, X_credit, 'creditcard_fraud_rf')

print("Model explainability completed and plots saved in the 'explainability' folder.")
