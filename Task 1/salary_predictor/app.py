from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the preprocessor and model
try:
    preprocessor = load('preprocessor.joblib')
except Exception as e:
    print("Error loading preprocessor:", e)
    preprocessor = None

try:
    model = load('best_salary_predictor_model (1).joblib')
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if preprocessor is None or model is None:
        return jsonify(error="Model not loaded")
    
    try:
        # Extract the features from the request
        data = request.get_json(force=True)
        
        # Extract specified inputs
        specified_inputs = {
            'SEX', 'AGE', 'PAST EXP', 'LEAVES USED', 'LEAVES REMAINING',
            'RATINGS', 'YEARS IN COMPANY', 'DESIGNATION', 'UNIT'
        }
        
        # Create DataFrame with specified inputs
        df = pd.DataFrame({key: [data[key]] for key in specified_inputs})
        
        # Preprocess the features
        processed_features = preprocessor.transform(df)
        
        # Make the prediction
        prediction = model.predict(processed_features)
        
        # Return the prediction as JSON
        return jsonify(result=prediction[0])
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
