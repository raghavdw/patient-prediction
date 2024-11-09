import joblib
import gradio as gr
import prometheus_client
from prometheus_client import Counter
from xgboost import XGBClassifier
import pandas as pd
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Load the trained model
model = joblib.load("model.joblib")

# Prometheus metric counters
prediction_counter = Counter("predictions_total", "Number of predictions made", ["outcome"])

# Define prediction function with all required inputs
def predict_survival(age, anaemia, creatinine_phosphokinase, diabetes, 
                     ejection_fraction, high_blood_pressure, platelets, 
                     serum_creatinine, serum_sodium, sex, smoking, time):
    # Organize inputs into the expected format for the model
    features = [[age, anaemia, creatinine_phosphokinase, diabetes, 
                 ejection_fraction, high_blood_pressure, platelets, 
                 serum_creatinine, serum_sodium, sex, smoking, time]]
    prediction = model.predict(features)
    return "Survived" if prediction[0] == 1 else "Not Survived"

# Gradio Interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Slider(40, 95, step=1, label="Age"),  # Range based on dataset
        gr.Dropdown(choices=[0, 1], label="Anaemia (0: No, 1: Yes)"),
        gr.Slider(20, 8000, step=1, label="Creatinine Phosphokinase"),  # Range based on dataset
        gr.Dropdown(choices=[0, 1], label="Diabetes (0: No, 1: Yes)"),
        gr.Slider(14, 80, step=1, label="Ejection Fraction"),  # Range based on dataset
        gr.Dropdown(choices=[0, 1], label="High Blood Pressure (0: No, 1: Yes)"),
        gr.Slider(47000, 800000, step=1000, label="Platelets"),  # Scaled for readability
        gr.Slider(0.6, 9.4, step=0.1, label="Serum Creatinine"),  # Range based on dataset
        gr.Slider(113, 146, step=1, label="Serum Sodium"),  # Range based on dataset
        gr.Dropdown(choices=[0, 1], label="Sex (0: Female, 1: Male)"),
        gr.Dropdown(choices=[0, 1], label="Smoking (0: No, 1: Yes)"),
        gr.Slider(4, 300, step=1, label="Time")  # Range based on dataset
    ],
    outputs="text",
    title="Patient Survival Prediction",
    description="Predicts if a patient will survive based on health metrics."
)

# Add Prometheus metrics endpoint
def metrics():
    return prometheus_client.generate_latest(), 200, {'Content-Type': 'text/plain'}

# Flask app for integrating Prometheus metrics
app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app,
    {"/metrics": metrics}  # Mounting the metrics endpoint
)

if __name__ == "__main__":
    # Launch Gradio and Flask together
    iface.launch(server_name="0.0.0.0", server_port=7860)
