from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #inputs from the website form
    tenure = float(request.form['tenure'])
    monthly = float(request.form['monthly'])
    
    #empty row with the right columns
    input_row = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    #specific inputs
    input_row['tenure'] = tenure
    input_row['MonthlyCharges'] = monthly

    #Scale
    scaled_input = scaler.transform(input_row)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    #result
    status = "LEAVING" if prediction == 1 else "STAYING"
    return render_template('index.html', 
                           output=status, 
                           prob=f"{round(probability*100, 2)}%")

if __name__ == "__main__":
    app.run(debug=True)