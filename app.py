import os
import gradio as gr
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

# --- KERAS 3 COMPATIBILITY PATCH ---
# This fixes the "TypeError: int() argument must be a string... not 'list'" error
# and handles version mismatches for BatchNormalization layers
from tensorflow.keras.layers import BatchNormalization
original_bn_init = BatchNormalization.__init__
def patched_bn_init(self, *args, **kwargs):
    if 'axis' in kwargs and isinstance(kwargs['axis'], list):
        kwargs['axis'] = kwargs['axis'][0]
    return original_bn_init(self, *args, **kwargs)
BatchNormalization.__init__ = patched_bn_init
# -----------------------------------

# Initialize Logging
print("--- STARTING APP INITIALIZATION ---")
print("Importing libraries...")

def load_artifacts(model_dir='models'):
    """Load model and preprocessing objects."""
    # Load preprocessing objects
    preprocessors_path = os.path.join(model_dir, 'preprocessing.pkl')
    with open(preprocessors_path, 'rb') as f:
        preprocessors = pickle.load(f)
    
    scaler = preprocessors['scaler']
    disease_mapping = preprocessors['disease_mapping']
    feature_names = preprocessors['feature_names']
    
    # Load model
    model_path = os.path.join(model_dir, 'finetuned_model.keras')
    print(f"Loading model from: {model_path}...")
    
    # CRITICAL: We load with compile=False to bypass optimizer version conflicts
    # Since we only use the model for predict(), we don't need the optimizer state.
    model = keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    
    return model, scaler, disease_mapping, feature_names

# Load model once at startup
print("Loading artifacts (this may take a minute on a cold start)...")
MODEL, SCALER, DISEASE_MAPPING, FEATURE_NAMES = load_artifacts()
print("Artifacts loaded. Feature Names:", FEATURE_NAMES)

def predict_risk(age, gender, family_history, hemoglobin, fetal_hemo, rdw, ferritin, brca1, p53, sweat, sickled, il6):
    # Map friendly labels back to encoded numeric values
    gender_map = {"Female": 0, "Male": 1}
    history_map = {"No": 0, "Yes": 1}
    
    # Create input dictionary
    data = {
        'Age': float(age),
        'Gender': gender_map[gender],
        'Family_History': history_map[family_history],
        'Hemoglobin': float(hemoglobin),
        'Fetal_Hemoglobin': float(fetal_hemo),
        'RDW_CV': float(rdw),
        'Serum_Ferritin': float(ferritin),
        'BRCA1_Expression': float(brca1),
        'p53_Mutation': float(p53),
        'Sweat_Chloride': float(sweat),
        'Sickled_RBC_Percent': float(sickled),
        'IL6_Level': float(il6)
    }
    
    # Create DataFrame and scale
    df = pd.DataFrame([data])[FEATURE_NAMES]
    X = SCALER.transform(df)
    
    # Predict
    preds = MODEL.predict(X, verbose=0)[0]
    
    # Format results
    results = {DISEASE_MAPPING[i]: float(preds[i]) for i in range(len(DISEASE_MAPPING))}
    
    # Add Healthy / Low Risk detection for low-confidence cases
    max_conf = np.max(preds)
    if max_conf < 0.60:
        results["Healthy / Low Risk"] = 1.0 - max_conf
    else:
        results["Healthy / Low Risk"] = 0.0
        
    return results

# Define Gradio Interface
interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(10, 80, value=30, label="Age"),
        gr.Radio(["Female", "Male"], label="Gender", value="Female"),
        gr.Radio(["No", "Yes"], label="Family History (Genetic Conditions)", value="No"),
        gr.Slider(5.0, 15.0, value=9.0, label="Hemoglobin Level (g/dL)"),
        gr.Slider(5.0, 20.0, value=10.0, label="Fetal Hemoglobin (%)"),
        gr.Slider(10.0, 25.0, value=15.0, label="RDW_CV (Red Cell Distribution Width)"),
        gr.Slider(10.0, 100.0, value=50.0, label="Serum Ferritin (ng/mL)"),
        gr.Slider(0.0, 1.0, value=0.3, label="BRCA1 Expression Level"),
        gr.Slider(0.0, 1.0, value=0.1, label="p53 Mutation Indicator"),
        gr.Slider(10.0, 120.0, value=40.0, label="Sweat Chloride (mmol/L)"),
        gr.Slider(0.0, 5.0, value=1.0, label="Sickled RBC Percent (%)"),
        gr.Slider(0.0, 20.0, value=5.0, label="IL6 Level (pg/mL)")
    ],
    outputs=gr.Label(num_top_classes=3, label="Predicted Genetic Disorder Risk"),
    title="🧬 Genetic Disease Prediction AI",
    description="This tool uses a Deep Forward Neural Network to predict the risk of genetic diseases (Thalassemia, Hemophilia, etc.) based on patient age and specific medical markers."
)

if __name__ == "__main__":
    # Use standard launch for Hugging Face to handle binding automatically
    interface.launch()
