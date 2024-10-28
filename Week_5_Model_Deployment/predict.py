import pickle
import os
import requests
from flask import Flask, request, jsonify

app = Flask('churn')

# Function to load the model and dictionary vectorizer
def load_model(C, n_splits):
    model_file = f'model_C={C}_splits={n_splits}.bin'
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file '{model_file}' not found.")
    with open(model_file, 'rb') as f_in:
        return pickle.load(f_in)

# Home route to confirm API is running
@app.route('/')
def home():
    return "Welcome to the Churn Prediction API"

# Prediction route that takes customer data as input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve JSON data from the POST request
        customer = request.get_json()
        if not customer:
            return jsonify({"error": "No data provided"}), 400
            
        # Check for missing required fields
        required_fields = ['tenure', 'monthlycharges', 'totalcharges']
        missing_fields = [field for field in required_fields if field not in customer]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        # Set model parameters with defaults if not provided
        C = customer.pop('C', int(1))
        n_splits = customer.pop('n_splits', 5)
        
        # Load model and dictionary vectorizer
        dv, model = load_model(C, n_splits)

        # Transform input data and make prediction
        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5

        # Prepare and return the result
        result = {
            "customer_id": customer.get('customer_id', 'unknown'),
            "churn_probability": float(y_pred),
            "churn": bool(churn),
            "model_parameters": {'C': C, 'n_splits': n_splits}
        }
        return jsonify(result)

    except Exception as e:
        # General exception handling to catch and return errors
        return jsonify({"error": str(e)}), 500

# Test endpoint to send a request to the predict endpoint
@app.route('/test', methods=['POST'])
def test():
    # URL for the local API prediction endpoint
    url = 'http://localhost:5001/predict'
    
    # Sample customer data
    customer = {
        "customer_id": "xyz-123",
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "phoneservice": "no",
        "multiplelines": "no_phone_service",
        "internetservice": "dsl",
        "onlinesecurity": "no",
        "onlinebackup": "yes",
        "deviceprotection": "no",
        "techsupport": "no",
        "streamingtv": "no",
        "streamingmovies": "no",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "tenure": 24,
        "monthlycharges": 29.85,
        "totalcharges": (24*29.85)
    }
    
    # Make POST request and return response
    response = requests.post(url, json=customer)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
