from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained models
with open('naive_bayes_model.pkl', 'rb') as nb_file:
    naive_bayes_model = pickle.load(nb_file)

with open('perceptron_model.pkl', 'rb') as perc_file:
    perceptron_model = pickle.load(perc_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'naive_bayes')
        
        # Convert input values to numeric types
        age = float(data['age'])
        glucose = float(data['glucose'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        
        input_features = np.array([[age, glucose, insulin, bmi]])

        if model_type == 'naive_bayes':
            prediction = naive_bayes_model.predict(input_features)
        elif model_type == 'perceptron':
            prediction = perceptron_model.predict(input_features)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        return jsonify({'diabetes_type': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)