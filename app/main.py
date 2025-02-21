from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import plotly.express as px
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, accuracy_score
import plotly.graph_objs as go
import plotly.io as pio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'data/NeighborhoodData_finalData_2.csv')
model_path = os.path.join(BASE_DIR, 'models/xgboost_model.pkl')


app = Flask(__name__)
app.secret_key = "your_secret_key"
# dataset_path = 'app/data/NeighborhoodData_finalData_2.csv'
# model_path = 'app/models/xgboost_model.pkl'
# Load your trained XGBoost model
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

# Dummy data for predictions and true labels
# Replace these with actual values from your dataset
y_true = [1, 0, 1, 1, 0]  # example true labels
y_pred_proba = [0.8, 0.3, 0.6, 0.9, 0.2]  # example prediction probabilities
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
# Load dataset on app start
print(os.path.exists(dataset_path))
data = pd.read_csv(dataset_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/visualize')
def visualize():
    graphs = []
    scatter_features = [
        ('medhhinc', 'obese_adult'), 
        ('medhhinc', 'unemp'), 
        ('unemp', 'birth_lw'),
        ('medhhinc', 'ed_hsgrad'), 
        ('chldfpl185', 'birth_lw'), 
        ('medhhinc', 'overobese_teen')
    ]
    for x, y in scatter_features:
        fig = px.scatter(data, x=x, y=y, color="Health_outcome", title=f"{x} vs {y}")
        graphs.append(fig.to_json())
        
    hist_features = ['medhhinc', 'obese_adult', 'unemp', 'birth_lw', 'ed_hsgrad', 'chldfpl185']
    for feature in hist_features:
        fig = px.histogram(data, x=feature, title=f"Distribution of {feature}", marginal="box")
        graphs.append(fig.to_json())

    box_features = ['medhhinc', 'obese_adult', 'unemp', 'birth_lw', 'ed_hsgrad', 'chldfpl185']
    for feature in box_features:
        fig = px.box(data, x="Health_outcome", y=feature, title=f"{feature} by Health Outcome")
        graphs.append(fig.to_json())

    fig = px.violin(data, x="Health_outcome", y="unemp", color="Health_outcome", title="Unemployment Rates by Health Outcome")
    graphs.append(fig.to_json())
    
    fig = px.violin(data, x="Health_outcome", y="birth_lw", color="Health_outcome", title="Low Birth Weight Rates by Health Outcome")
    graphs.append(fig.to_json())

    fig = px.bar(data, x="Health_outcome", y="ed_hsgrad", color="Health_outcome", title="High School Graduation Rates by Health Outcome")
    graphs.append(fig.to_json())
    
    fig = px.bar(data, x="Health_outcome", y="famfpl185", color="Health_outcome", title="Percentage of Families Below Poverty Line by Health Outcome")
    graphs.append(fig.to_json())

    return render_template('visualization.html', graphs=graphs)

@app.route('/metricReport')
def metric_report():
    # Convert Plotly figures to JSON for embedding in HTML
    roc_auc_plot = pio.read_json("static/plots_metrics/roc_auc_plot.json")
    conf_matrix_plot = pio.read_json("static/plots_metrics/conf_matrix_plot.json")
    metrics_plot = pio.read_json("static/plots_metrics/metrics_plot.json")
    
    # Pass data to template as JSON strings
    return render_template('metricReport.html', 
                           roc_auc_plot=roc_auc_plot.to_json(),
                           conf_matrix_plot=conf_matrix_plot.to_json(),
                           metrics_plot=metrics_plot.to_json())

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction ='Prediction results will appear here'
    if request.method == 'POST':
            user_data = request.form.to_dict(flat=True)
            prediction = predict_health_outcome(user_data, model)
            # return redirect(url_for('predict'))
    return render_template('predict.html', prediction=prediction)
    
def predict_health_outcome(user_input, model):
    """
    Predicts the Health Outcome based on user input.
    
    Parameters:
    - user_input (dict): A dictionary of feature values for prediction.
    - model: Trained model pipeline (default is best estimator from grid search).
    
    Returns:
    - Prediction result as a class label.
    """
    # Convert user input to DataFrame for consistency in preprocessing
    input_df = pd.DataFrame([user_input])
    
    # Predict using the trained pipeline
    prediction = model.predict(input_df)
    
    # Map encoded prediction to actual class name if encoding was applied
    health_outcome_mapping = {0: "At Risk", 1: "Chronic", 2: "Healthy"}
    prediction_label = health_outcome_mapping[prediction[0]]
    
    return prediction_label

# def metric_report():
#     """
#     Calculates and returns performance metrics for the trained model.
    
#     Returns:
#     - Dictionary containing ROC AUC, Confusion Matrix, Precision, Recall, and Accuracy.
#     """
#     # Calculate metrics
#     # create train test split
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Predict on test set
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     roc_auc = roc_auc_score(y_true, y_pred_proba)
#     conf_matrix = confusion_matrix(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    app.run(debug=True)
