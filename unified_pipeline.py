#!/usr/bin/env python3
"""
Genetic Disease Risk Prediction - Unified Pipeline
This script combines all logic (data loading, preprocessing, model definition, and training)
into a single file for an end-to-end flow.
"""

import os
import sys
import json
import shutil
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from datetime import datetime
from itertools import cycle
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# =============================================================================
# 1. UTILITIES
# =============================================================================

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_gpu():
    """Configure GPU settings for TensorFlow."""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"GPU setup error: {e}")

# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================

def download_kaggle_dataset(dataset_name='syeddanish5/genetic-disease-prediction-dataset', output_dir='data'):
    """Download dataset from Kaggle using kaggle CLI."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        kaggle_cmd = "kaggle"
        if shutil.which(kaggle_cmd) is None:
            possible_paths = [
                os.path.expanduser("~/Library/Python/3.9/bin/kaggle"),
                os.path.expanduser("~/.local/bin/kaggle"),
                os.path.join(sys.prefix, "bin", "kaggle")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    kaggle_cmd = path
                    break
        
        cmd = [kaggle_cmd, "datasets", "download", "-d", dataset_name, "-p", output_dir, "--unzip"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_kaggle_dataset(data_path='data/genetic_disease.csv', target_column='Disease'):
    """Load and preprocess the Kaggle Genetic Disease Prediction dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
    
    if target_column not in df.columns:
        possible_targets = ['Disease', 'disease', 'Target', 'Label']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
    
    y_raw = df[target_column]
    X_df = df.drop(columns=[target_column])

    # Map numeric labels if needed
    if pd.api.types.is_numeric_dtype(y_raw) and set(y_raw.unique()).issubset({0, 1, 2, 3, 4}):
        disease_name_map = {0: 'Thalassemia', 1: 'Hemophilia', 2: 'Breast Cancer', 3: 'Sickle Cell Anemia', 4: 'Cystic Fibrosis'}
        y_raw = y_raw.map(disease_name_map)
    
    # Handle categorical features
    label_encoders = {}
    for col in X_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le
    
    X_df = X_df.fillna(X_df.median())
    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()
    
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    disease_mapping = dict(zip(range(len(target_encoder.classes_)), target_encoder.classes_))
    
    return X, y, feature_names, disease_mapping, label_encoders

def prepare_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split and scale data."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test, 'scaler': scaler}

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

def create_model(input_dim, hidden_layers=[512, 256, 128, 64], dropout_rates=[0.5, 0.4, 0.3, 0.2], l2_reg=0.001, learning_rate=0.001, num_classes=5):
    """Create the neural network model."""
    model = keras.Sequential(name='GeneticDiseasePredictor')
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    
    for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg), name=f'dense_{i+1}'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class GeneticDiseaseModelWrapper:
    """Wrapper for the Keras model."""
    def __init__(self, input_dim, num_classes=5, learning_rate=0.001):
        self.model = create_model(input_dim, num_classes=num_classes, learning_rate=learning_rate)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_training_results(history, save_dir='visualizations'):
    """Plot accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.legend()
    
    plt.savefig(f'{save_dir}/learning_curves.png')
    plt.close()

def plot_evaluation_metrics(y_test, y_pred_probs, disease_mapping, save_dir='visualizations'):
    """Plot ROC curves and Confusion Matrix."""
    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(disease_mapping)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{disease_mapping[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curves.png')
    plt.close()
    
    # Confusion Matrix
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(disease_mapping.values()), 
                yticklabels=list(disease_mapping.values()))
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

# =============================================================================
# 5. MAIN FLOW
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Genetic Disease Prediction Pipeline')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='models', help='Model save directory')
    args = parser.parse_args()

    set_random_seed(42)
    setup_gpu()

    # 1. Load Data
    print("\n--- Step 1: Data Loading ---")
    os.makedirs(args.data_dir, exist_ok=True)
    data_file = os.path.join(args.data_dir, 'genetic_disease.csv')
    
    if not os.path.exists(data_file):
        # Check for any CSV in the data directory
        csv_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        if csv_files:
            data_file = os.path.join(args.data_dir, csv_files[0])
        else:
            print("Data not found. Downloading...")
            success = download_kaggle_dataset(output_dir=args.data_dir)
            if success:
                csv_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
                if csv_files:
                    data_file = os.path.join(args.data_dir, csv_files[0])
                else:
                    print("Error: No CSV file found after download.")
                    return
            else:
                print("Error: Failed to download dataset.")
                return
    
    print(f"Using data file: {data_file}")
    X, y, feature_names, disease_mapping, encoders = load_kaggle_dataset(data_file)
    splits = prepare_data(X, y)
    
    # Save preprocessing
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f'{args.save_dir}/preprocessing.pkl', 'wb') as f:
        pickle.dump({'scaler': splits['scaler'], 'encoders': encoders, 'mapping': disease_mapping}, f)

    # 2. Build and Train Model
    print("\n--- Step 2: Model Training ---")
    wrapper = GeneticDiseaseModelWrapper(input_dim=X.shape[1], num_classes=len(disease_mapping))
    
    model_path = f'{args.save_dir}/unified_model.keras'
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = wrapper.train(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'], 
                           epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    # 3. Evaluation and Visualization
    print("\n--- Step 3: Evaluation ---")
    plot_training_results(history)
    
    y_pred_probs = wrapper.model.predict(splits['X_test'])
    plot_evaluation_metrics(splits['y_test'], y_pred_probs, disease_mapping)
    
    accuracy = accuracy_score(splits['y_test'], np.argmax(y_pred_probs, axis=1))
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Workflow complete. Results saved in 'models/' and 'visualizations/'.")

if __name__ == "__main__":
    main()
