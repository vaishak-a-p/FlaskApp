from flask import Flask, render_template, request, flash
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# loading the ml model
lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# loading the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.values['gender'])
    age = int(request.values['age'])
    salary = int(request.values['salary'])
    X_new = pd.DataFrame({'Age': [age], 'EstimatedSalary': [
                         salary], 'Gender_Male': [gender]})
    X_new_transformed = scaler.transform(X_new)
    out = lr_model.predict(X_new_transformed)
    print(f"Output:{out}")
    return render_template('home.html', prediction_text=f'Plan purchased: {"Yes" if out[0] else "No"}')


if __name__ == "__main__":
    app.run()
