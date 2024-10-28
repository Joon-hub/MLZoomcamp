import pickle
import os
from flask import Flask, request, jsonify

def load_model(C, n_splits):
    model_file = f'model_C={C}_splits={n_splits}.bin'
    if not os.path.exists(model_file):
        return None, None

    with open(model_file, 'rb') as f_in:
        return pickle.load(f_in)

app = Flask('churn')
@app.route('/')
def home():
    return "Welcome to the Churn Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    C = customer.pop('C', 1.0)              # Extract C parameter from the request, default 1.0 
    n_splits = customer.pop('n_splits', 5)  # default to 5 if not provided
    
    dv, model = load_model(C, n_splits)
    
    if dv is None or model is None:
        return jsonify({"error": f"Model with C={C} and n_splits={n_splits} not found"}), 404

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn),
        'model_parameter_C': C,
        'model_parameter_n_splits': n_splits
    }

    return jsonify(result)

@app.route('/models', methods=['GET'])
def list_models():
    models = [f.replace('model_C=', '').replace('.bin', '') 
              for f in os.listdir('.') if f.startswith('model_C=') and f.endswith('.bin')]
    return jsonify({"available_models": models})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)