from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") #Rectify Warning (DOESN'T AFFECT RESULTS SO I SUPPRESSED IT)

class CustomPerceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                if y_[idx] * y_predicted <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

app = Flask(__name__)
CORS(app)

with open('naive_bayes_model.pkl', 'rb') as nb_file:
    naive_bayes_model = pickle.load(nb_file)

with open('perceptron_model.pkl', 'rb') as perc_file:
    perceptron_model = pickle.load(perc_file)

with open('custom_perceptron_model.pkl', 'rb') as custom_perc_file:
    custom_perceptron_model = pickle.load(custom_perc_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'naive_bayes')
        age = float(data['age'])
        glucose = float(data['glucose'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        
        input_features = pd.DataFrame([[glucose, insulin, bmi, age]], columns=['Glucose', 'Insulin', 'BMI', 'Age'])
        input_features_scaled = scaler.transform(input_features)

        if model_type == 'naive_bayes':
            prediction = naive_bayes_model.predict(input_features_scaled)
        elif model_type == 'perceptron':
            prediction = perceptron_model.predict(input_features_scaled)
        elif model_type == 'custom_perceptron':
            prediction = custom_perceptron_model.predict(input_features_scaled)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        return jsonify({'diabetes_type': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)