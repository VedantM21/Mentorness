from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
lr_model = joblib.load('logistic_regression_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        transaction_data = request.form.to_dict()
        print("Received transaction data:", transaction_data)
        
        # Convert data to DataFrame
        transaction_df = pd.DataFrame([transaction_data])
        print("DataFrame from transaction data:")
        print(transaction_df)
        
        # Preprocess the data
        transaction_df['Timestamp'] = pd.to_datetime(transaction_df['Timestamp'])
        transaction_df['Transaction_Amount'] = pd.to_numeric(transaction_df['Transaction_Amount'], errors='coerce')
        transaction_df['Amount_paid'] = pd.to_numeric(transaction_df['Amount_paid'], errors='coerce')
        transaction_df['Vehicle_Speed'] = pd.to_numeric(transaction_df['Vehicle_Speed'], errors='coerce')
        transaction_df['Hour'] = transaction_df['Timestamp'].dt.hour
        transaction_df['DayOfWeek'] = transaction_df['Timestamp'].dt.dayofweek
        transaction_df['IsWeekend'] = transaction_df['DayOfWeek'].isin([5, 6]).astype(int)
        
        columns_to_drop = ['Transaction_ID', 'FastagID', 'TollBoothID', 'Vehicle_Plate_Number']
        transaction_df = transaction_df.drop(columns=columns_to_drop, errors='ignore')
        print("Processed DataFrame:")
        print(transaction_df)
        
        # Make prediction using the loaded model
        prediction = lr_model.predict(transaction_df)
        print("Prediction:", prediction)
        print("Prediction type:", type(prediction[0]))

        # Check the type and handle accordingly
        if isinstance(prediction[0], str):
            prediction_label = prediction[0]
        else:
            fraud_labels = {1: 'Fraud', 0: 'Not Fraud'}
            prediction_label = fraud_labels.get(prediction[0], 'Unknown')
        
        return render_template('index.html', prediction=prediction_label)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
