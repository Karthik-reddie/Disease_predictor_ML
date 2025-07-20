import pandas as pd
import joblib
import gradio as gr

# Load the model
model = joblib.load("disease_model.pkl")  # We'll export this model later

# Prediction function
def predict(age, temperature, symptoms):
    input_data = pd.DataFrame([[age, temperature, symptoms]],
                              columns=["age", "temperature", "symptoms"])
    prediction = model.predict(input_data)
    return prediction[0]

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Temperature"),
        gr.Textbox(label="Symptoms")
    ],
    outputs="text",
    title="Disease Predictor"
)

iface.launch()
