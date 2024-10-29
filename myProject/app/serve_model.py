import os
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask application
app = Flask(__name__)

# Determine the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models using absolute paths
fraud_rf_model_path = os.path.join(base_dir, 'models', 'fraud_detection_rf_model.pkl')
fraud_dt_model_path = os.path.join(base_dir, 'models', 'fraud_detection_dt_model.pkl')
creditcard_rf_model_path = os.path.join(base_dir, 'models', 'creditcard_fraud_rf_model.pkl')
creditcard_dt_model_path = os.path.join(base_dir, 'models', 'creditcard_fraud_dt_model.pkl')

# Load the models and handle any potential errors
try:
    fraud_rf_model = joblib.load(fraud_rf_model_path)
    fraud_dt_model = joblib.load(fraud_dt_model_path)
    creditcard_rf_model = joblib.load(creditcard_rf_model_path)
    creditcard_dt_model = joblib.load(creditcard_dt_model_path)
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}")
    raise

# Define the home route to serve an HTML interface (if using one)
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for Random Forest fraud data predictions
@app.route('/predict_fraud_rf', methods=['POST'])
def predict_fraud_rf():
    try:
        data = request.get_json()
        features = [
            data['purchase_value'],
            data['transaction_freq'],
            data['transaction_velocity'],
            data['hour_of_day'],
            data['day_of_week'],
            data['source_encoded'],
            data['browser_encoded'],
            data['sex_encoded']
        ]
        
        # Make prediction
        prediction = fraud_rf_model.predict([features])
        logging.info("Fraud RF prediction made.")
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logging.error(f"Error in /predict_fraud_rf: {e}")
        return jsonify({'error': str(e)}), 500

# API endpoint for Decision Tree fraud data predictions
@app.route('/predict_fraud_dt', methods=['POST'])
def predict_fraud_dt():
    try:
        data = request.get_json()
        features = [
            data['purchase_value'],
            data['transaction_freq'],
            data['transaction_velocity'],
            data['hour_of_day'],
            data['day_of_week'],
            data['source_encoded'],
            data['browser_encoded'],
            data['sex_encoded']
        ]
        
        # Make prediction
        prediction = fraud_dt_model.predict([features])
        logging.info("Fraud DT prediction made.")
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logging.error(f"Error in /predict_fraud_dt: {e}")
        return jsonify({'error': str(e)}), 500

# API endpoint for Random Forest credit card fraud predictions
@app.route('/predict_creditcard_rf', methods=['POST'])
def predict_creditcard_rf():
    try:
        data = request.get_json()
        features = np.array([data[f'V{i}'] for i in range(1, 29)] + [data['Amount']])
        
        # Make prediction
        prediction = creditcard_rf_model.predict([features])
        logging.info("Credit card RF prediction made.")
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logging.error(f"Error in /predict_creditcard_rf: {e}")
        return jsonify({'error': str(e)}), 500

# API endpoint for Decision Tree credit card fraud predictions
@app.route('/predict_creditcard_dt', methods=['POST'])
def predict_creditcard_dt():
    try:
        data = request.get_json()
        features = np.array([data[f'V{i}'] for i in range(1, 29)] + [data['Amount']])
        
        # Make prediction
        prediction = creditcard_dt_model.predict([features])
        logging.info("Credit card DT prediction made.")
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logging.error(f"Error in /predict_creditcard_dt: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
