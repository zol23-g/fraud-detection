from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Load the fraud detection models
fraud_model_path = os.path.join('notebooks/models/fraud_detection_rf_model.pkl')
creditcard_model_path = os.path.join('notebooks/models/creditcard_fraud_rf_model.pkl')

with open(fraud_model_path, 'rb') as f:
    fraud_model = pickle.load(f)

with open(creditcard_model_path, 'rb') as f:
    creditcard_model = pickle.load(f)

# API endpoint for fraud data predictions (Fraud_Data.csv)
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    
    # Extract input features (must match the training input structure)
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
    
    # Make prediction using the loaded fraud model
    prediction = fraud_model.predict([features])
    
    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

# API endpoint for credit card fraud detection (creditcard.csv)
@app.route('/predict_creditcard', methods=['POST'])
def predict_creditcard():
    data = request.get_json()
    
    # Extract input features for credit card fraud model (must match input structure)
    features = np.array([data[f'V{i}'] for i in range(1, 29)] + [data['Amount']])
    
    # Make prediction using the loaded credit card fraud model
    prediction = creditcard_model.predict([features])
    
    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
