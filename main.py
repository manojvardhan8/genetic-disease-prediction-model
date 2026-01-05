#!/usr/bin/env python3
"""
Genetic Disease Risk Prediction using Kaggle Dataset

Train the model using the Kaggle Genetic Disease Prediction dataset.
Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/genetic-disease-prediction-dataset

Usage:
    python main_kaggle.py [options]

Examples:
    python main_kaggle.py --epochs 50
    python main_kaggle.py --data-path data/genetic_disease.csv --epochs 100
"""

import argparse
import os
import sys
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from src.kaggle_data_loader import (
    load_or_download_dataset, 
    load_kaggle_dataset,
    prepare_data_for_multiclass,
    save_preprocessing_objects
)
from src.feature_selection import FeatureSelector
from src.model import GeneticDiseaseModel, get_callbacks
from src.utils import set_random_seed, setup_gpu


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Genetic Disease Prediction using Kaggle Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to CSV file (optional, will auto-detect)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files (default: data)')

    
    # Feature selection
    parser.add_argument('--feature-selection', type=str, default=None,
                       choices=['variance', 'chi2', 'mutual_info', 'rfe', 'lasso', 'random_forest'],
                       help='Feature selection method (optional)')
    parser.add_argument('--num-features', type=int, default=None,
                       help='Number of features to select (optional)')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='default',
                       choices=['small', 'default', 'large'],
                       help='Model size (default: default)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup GPU
    setup_gpu()
    
    print("\n" + "#"*70)
    print("#" + " "*18 + "GENETIC DISEASE PREDICTION" + " "*21 + "#")
    print("#" + " "*12 + "Using Kaggle Genetic Disease Dataset" + " "*14 + "#")
    print("#"*70 + "\n")
    

    
    # Load data
    print("="*60)
    print("Step 1: Loading Kaggle Dataset")
    print("="*60)
    
    try:
        if args.data_path:
            X, y, feature_names, disease_mapping, label_encoders = load_kaggle_dataset(args.data_path)
        else:
            X, y, feature_names, disease_mapping, label_encoders = load_or_download_dataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nERROR: Dataset not found and could not be downloaded.")
        print("Please configure Kaggle API credentials or download the dataset manually.")
        sys.exit(1)
    
    num_classes = len(disease_mapping)
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")
    
    # Feature selection (optional)
    if args.feature_selection and args.num_features:
        print("\n" + "="*60)
        print("Step 2: Feature Selection")
        print("="*60)
        
        fs = FeatureSelector(random_state=args.seed)
        X = fs.select_features(X, y, method=args.feature_selection, k=args.num_features)
        print(f"Selected {X.shape[1]} features using {args.feature_selection}")
        feature_names = [feature_names[i] for i in fs.get_selected_feature_indices()]
    
    # Prepare data splits
    print("\n" + "="*60)
    print("Step 3: Preparing Data Splits")
    print("="*60)
    
    splits = prepare_data_for_multiclass(X, y, random_state=args.seed)
    print(f"Training set: {splits['X_train'].shape}")
    print(f"Validation set: {splits['X_val'].shape}")
    print(f"Test set: {splits['X_test'].shape}")
    # Save preprocessing objects
    print(f"Saving preprocessing objects to {args.save_dir}...")
    save_preprocessing_objects(
        args.save_dir,
        splits['scaler'],
        label_encoders,
        disease_mapping,
        feature_names
    )
    
    # Build model
    print("\n" + "="*60)
    print("Step 4: Building Model")
    print("="*60)
    
    input_dim = splits['X_train'].shape[1]
    
    model = GeneticDiseaseModel(
        input_dim=input_dim,
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        num_classes=num_classes
    )
    
    print(f"Model size: {args.model_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Output classes: {num_classes}")
    model.summary()
    
    # Train model
    print("\n" + "="*60)
    print("Step 5: Training Model")
    print("="*60)
    
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'kaggle_model.keras')
    
    callbacks = get_callbacks(model_path=model_path, patience=10)
    
    print(f"Training for up to {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    
    history = model.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Step 6: Evaluating on Test Set")
    print("="*60)
    
    # Make predictions
    y_pred = model.model.predict(splits['X_test'])
    if num_classes > 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == splits['y_test'])
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class Results:")
    for class_idx, class_name in disease_mapping.items():
        mask = splits['y_test'] == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == splits['y_test'][mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nClassification Report:")
    print("-"*50)
    target_names = [disease_mapping[i] for i in range(num_classes)]
    print(classification_report(splits['y_test'], y_pred_classes, target_names=target_names))
    
    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(splits['y_test'], y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('visualizations', exist_ok=True)
    save_path = 'visualizations/confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\nConfusion Matrix Heatmap saved to: {save_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"\nDisease Classes: {list(disease_mapping.values())}")


if __name__ == "__main__":
    main()
