# Diabetes Prediction Using Machine Learning

## Overview
This project implements a machine learning solution to predict diabetes using a dataset that includes features such as Glucose, Insulin, BMI, and Age. It utilizes two models, **Naive Bayes** and **Perceptron**, to demonstrate classification capabilities. The results can help understand the relationship between the input features and the likelihood of diabetes.

## Features
- **Two Machine Learning Models**: Implements Naive Bayes and Perceptron models for classification.
- **Data Preprocessing**: Handles missing values, scaling, and outlier detection.
- **Model Evaluation**: Provides accuracy, confusion matrix, and classification reports.
- **Flask API**: A RESTful API to serve predictions for user-provided input data.

## Requirements
- Python 3
- Flask
- scikit-learn
- pandas
- numpy
- requests

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/diabetes-prediction.git
2. Navigate to project directory
   ```bash
   cd diabetes-prediction
4. Create virtual environment (Optional)
   ```bash
   python -m venv venv
   source venv/bin/activate
5. Run ipynb file for model training on Google Collab or Jupyter Notebook
6. Start the Flask backend
   ```bash
   python3 app.py
   - The API will run on http://127.0.0.1:5000
