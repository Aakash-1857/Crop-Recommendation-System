from flask import Flask, request, render_template,jsonify
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load All The Models ---
print("--> Loading model...")
model = joblib.load('crop_model.pkl')
print("--> Loading scaler...")
scaler = joblib.load('scaler.pkl')
print("--> Loading column names...")
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)['columns']
print("--> All artifacts loaded successfully!")
# ---------------------------

@app.route('/')
def home():
    return render_template('index.html')

# In app.py

# --- Create the Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the data from the JSON sent by the fetch request
    data = request.get_json()
    
    # 2. Create an ordered list of the feature values
    # We use the loaded model_columns to ensure the order is correct
    input_data = [float(data[col]) for col in model_columns]
    
    # 3. Create a DataFrame for scaling
    input_df = pd.DataFrame([input_data], columns=model_columns)
    
    # 4. Scale the input data
    scaled_data = scaler.transform(input_df)
    
    # 5. Make a prediction
    prediction = model.predict(scaled_data)
    
    # 6. Return the result as a JSON object
    # This is what the frontend's fetch() call is expecting
    return jsonify({'result': prediction[0]})

# In app.py, add this new route

@app.route('/result')
def result_page():
    return render_template('result.html')
if __name__ == '__main__':
    app.run()