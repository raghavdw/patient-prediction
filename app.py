import joblib
import gradio as gr

# Load the trained model
model = joblib.load("model.joblib")

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
        gr.Number(label="Age"),
        gr.Number(label="Anaemia (0: No, 1: Yes)"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Number(label="Diabetes (0: No, 1: Yes)"),
        gr.Number(label="Ejection Fraction"),
        gr.Number(label="High Blood Pressure (0: No, 1: Yes)"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Number(label="Sex (0: Female, 1: Male)"),
        gr.Number(label="Smoking (0: No, 1: Yes)"),
        gr.Number(label="Time")
    ],
    outputs="text",
    title="Patient Survival Prediction",
    description="Predicts if a patient will survive based on health metrics."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
