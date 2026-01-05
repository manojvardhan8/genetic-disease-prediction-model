#!/usr/bin/env python3
"""
Prediction Script for Genetic Disease Risk

Loads the trained model and preprocessing objects to make predictions on new data.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from src.kaggle_data_loader import load_preprocessing_objects

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_artifacts(model_dir='models'):
    """Load model and preprocessing objects."""
    print(f"Loading artifacts from {model_dir}...")
    
    # Load preprocessing objects
    try:
        preprocessors = load_preprocessing_objects(model_dir)
        scaler = preprocessors['scaler']
        label_encoders = preprocessors['label_encoders']
        disease_mapping = preprocessors['disease_mapping']
        feature_names = preprocessors['feature_names']
    except FileNotFoundError:
        print("Error: Preprocessing objects not found.")
        print("Please run 'python3 main_kaggle.py' first to train the model and save artifacts.")
        exit(1)
        
    # Load model
    model_path = os.path.join(model_dir, 'kaggle_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)
        
    model = keras.models.load_model(model_path)
    
    return model, scaler, label_encoders, disease_mapping, feature_names


def preprocess_input(data_dict, scaler, label_encoders, feature_names):
    """
    Preprocess raw input data into model-ready format.
    """
    # Create DataFrame with correct column order
    df = pd.DataFrame([data_dict])
    
    # Ensure all features exist
    for feature in feature_names:
        if feature not in df.columns:
            raise ValueError(f"Missing feature: {feature}")
            
    # Select and order columns
    df = df[feature_names]
    
    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                # If label not seen during training, try to handle gracefully or error
                print(f"Warning: Unseen label in column {col}. Using most frequent.")
                # Fallback to first class (usually 0) or handle differently
                df[col] = 0
                
    # Scale features
    X = scaler.transform(df)
    
    return X


def predict_disease(model, X, disease_mapping):
    """Make prediction using the model."""
    predictions = model.predict(X, verbose=0)
    
    # Multi-class prediction
    if predictions.shape[1] > 1:
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        disease = disease_mapping[class_idx]
        
        # Get all probabilities
        probs = {disease_mapping[i]: float(predictions[0][i]) for i in range(len(disease_mapping))}
        
    else:
        # Binary prediction
        prob = predictions[0][0]
        class_idx = int(prob > 0.5)
        confidence = prob if class_idx == 1 else 1 - prob
        disease = disease_mapping[class_idx]
        probs = {disease_mapping[1]: float(prob), disease_mapping[0]: float(1-prob)}
        
    return disease, confidence, probs


def get_user_input(feature_names, label_encoders):
    """Interactive input from user."""
    print("\n--- Enter Patient Data ---")
    data = {}
    
    # Define some default/descriptions for known features to help user
    descriptions = {
        'Age': 'Age in years',
        'Gender': 'Male/Female',
        'Hemoglobin_Level': 'g/dL (e.g., 12.5)',
        'Family_History': 'Yes/No',
        # Add more as needed
    }
    
    for feature in feature_names:
        prompt = f"{feature}"
        if feature in descriptions:
            prompt += f" ({descriptions[feature]})"
        
        # Show options for categorical
        if feature in label_encoders:
            options = label_encoders[feature].classes_
            prompt += f" [{'/'.join(options)}]"
            
        value = input(f"{prompt}: ")
        data[feature] = value
        
    return data


def main():
    parser = argparse.ArgumentParser(description='Predict Genetic Disease Risk')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory with model and artifacts')
    parser.add_argument('--interactive', action='store_true', help='Input data interactively')
    args = parser.parse_args()
    
    # Load artifacts
    model, scaler, label_encoders, disease_mapping, feature_names = load_artifacts(args.model_dir)
    
    if args.interactive:
        data = get_user_input(feature_names, label_encoders)
    else:
        # data = get_sample_data() # Logic to get sample data if not interactive?
        # For now, let's use a hardcoded sample compatible with the Kaggle dataset
        print("\nUsing default sample data (Normal Male, 30yo)...")
        data = {
            'Age': 30,
            'Gender': 'Male',
            'Family_History': 'No',
            'Hemoglobin_Level': 13.5,
            'Fetal_Hemoglobin': 1.0,
            'RDW_CV': 13.0,
            'Serum_Ferritin': 100,
            'BRCA1_Expression': 2.5,
            'p53_Mutation': 0,
            'CFTR_Mutation': 0,
            'HBB_Gene_Variant': 0,
            'Factor_VIII_Level': 100,
            'White_Blood_Cell': 7.0,
            'Platelet_Count': 250,
            'Blood_Type': 'O'
        }
        # Check if we are missing any keys from feature_names
        missing = [f for f in feature_names if f not in data]
        if missing:
             # Fill missing with dummy values for demonstration
             print(f"Note: Filling missing features {missing} with defaults")
             for f in missing:
                 if f in label_encoders:
                     data[f] = label_encoders[f].classes_[0]
                 else:
                     data[f] = 0
    
    print("\nInput Data:")
    for k, v in data.items():
        print(f"  {k}: {v}")
        
    try:
        # Preprocess
        X = preprocess_input(data, scaler, label_encoders, feature_names)
        
        # Predict
        disease, confidence, probs = predict_disease(model, X, disease_mapping)
        
        print("\n" + "="*40)
        if confidence < 0.60:
            print(f"PREDICTION: No Disease / Healthy (Low Confidence)")
            print(f"Closest match: {disease} ({confidence:.2%})")
            print("\n*Note: The model is not confident in this prediction (<60%).")
            print("This suggests the patient may be Healthy or have an unknown condition.")
        else:
            print(f"PREDICTION: {disease}")
            print(f"Confidence: {confidence:.2%}")
        print("="*40)
        
        print("\nProbability Distribution:")
        for d, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {d}: {p:.2%}")
            
    except Exception as e:
        print(f"\nError predicting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
