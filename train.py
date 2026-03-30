#!/usr/bin/env python3
"""
Genetic Disease Risk Prediction - Unified Training Script

This script supports both base training and fine-tuning of the model.
Usage:
    Base training:    python3 train.py --epochs 7
    Fine-tuning:      python3 train.py --finetune --epochs 7
"""

import argparse
import os
import sys
import warnings
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from src.kaggle_data_loader import (
    load_or_download_dataset, 
    load_kaggle_dataset,
    prepare_data_for_multiclass,
    save_preprocessing_objects
)
from src.model import GeneticDiseaseModel, get_callbacks
from src.utils import set_random_seed, setup_gpu


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Genetic Disease Prediction - Unified Training Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode switch
    parser.add_argument('--finetune', action='store_true',
                       help='Perform fine-tuning on a pre-trained model')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to CSV file (optional)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files (default: data)')
    
    # Model/Training parameters
    parser.add_argument('--epochs', type=int, default=7,
                        help='Number of training epochs (default: 7)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (defaults: 0.001 for base, 0.00005 for fine-tune)')
    parser.add_argument('--model-size', type=str, default='default',
                       choices=['small', 'default', 'large'],
                       help='Model size for base training (default: default)')
    
    # Paths
    parser.add_argument('--load-model', type=str, default='models/kaggle_model.keras',
                       help='Path to load model for fine-tuning')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models and preprocessing objects')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


def save_history_json(history, save_path):
    """Save training history to a JSON file."""
    # Convert history object to serializable dict
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(save_path, 'w') as f:
        json.dump(history_dict, f)
    print(f"History saved to: {save_path}")


def load_history_json(load_path):
    """Load training history from a JSON file."""
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            return json.load(f)
    return None


def plot_history(history, is_finetune=False, base_history=None):
    """Plot training and validation history (Combined if possible)."""
    if is_finetune and base_history:
        # Combine base and fine-tune history
        fine_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        combined = {}
        for key in base_history.keys():
            if key in fine_dict:
                combined[key] = base_history[key] + fine_dict[key]
        
        transition_epoch = len(base_history.get('loss', []))
        epochs = range(len(combined.get('loss', [])))
        
        plt.figure(figsize=(15, 6))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, combined.get('accuracy'), label='Train Accuracy', lw=2)
        plt.plot(epochs, combined.get('val_accuracy'), label='Val Accuracy', lw=2)
        plt.axvline(x=transition_epoch - 0.5, color='r', linestyle='--', label='Fine-tuning Start')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, combined.get('loss'), label='Train Loss', lw=2)
        plt.plot(epochs, combined.get('val_loss'), label='Val Loss', lw=2)
        plt.axvline(x=transition_epoch - 0.5, color='r', linestyle='--', label='Fine-tuning Start')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs('visualizations', exist_ok=True)
        save_path = 'visualizations/combined_curves.png'
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nCombined training curves saved to: {save_path}")
        plt.close()
    else:
        # Standard plot for base training
        history_dict = history.history
        epochs = range(len(history_dict.get('loss')))
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict.get('accuracy'), label='Train Accuracy')
        plt.plot(epochs, history_dict.get('val_accuracy'), label='Val Accuracy')
        plt.title('Model Accuracy (Base)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict.get('loss'), label='Train Loss')
        plt.plot(epochs, history_dict.get('val_loss'), label='Val Loss')
        plt.title('Model Loss (Base)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/base_curves.png')
        plt.close()


def plot_roc_curves(y_test, y_pred_probs, disease_mapping, is_finetune=False):
    """Plot ROC curves for each class."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

    n_classes = len(disease_mapping)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    
    for i, color in zip(range(n_classes), colors):
        class_name = disease_mapping[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_name} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title_suffix = "(Fine-tuned)" if is_finetune else "(Base Model)"
    plt.title(f'ROC Curves {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs('visualizations', exist_ok=True)
    suffix = "finetuned" if is_finetune else "base"
    save_path = f'visualizations/{suffix}_roc_curve.png'
    plt.savefig(save_path)
    print(f"ROC curves saved to: {save_path}")
    plt.close()


def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    setup_gpu()
    
    # Set default learning rates if not provided
    if args.learning_rate is None:
        args.learning_rate = 0.001
    
    mode_text = "FINE-TUNING MODE" if args.finetune else "BASE TRAINING MODE"
    print("\n" + "#"*70)
    print("#" + " "*((70-len(mode_text)-2)//2) + mode_text + " "*((70-len(mode_text)-1)//2) + "#")
    print("#"*70 + "\n")
    
    # 1. Load data
    print("="*60)
    print("Step 1: Loading Dataset")
    print("="*60)
    
    try:
        if args.data_path:
            X, y, feature_names, disease_mapping, label_encoders = load_kaggle_dataset(args.data_path)
        else:
            X, y, feature_names, disease_mapping, label_encoders = load_or_download_dataset(args.data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    num_classes = len(disease_mapping)
    splits = prepare_data_for_multiclass(X, y, random_state=args.seed)
    
    # Save preprocessing objects
    os.makedirs(args.save_dir, exist_ok=True)
    save_preprocessing_objects(args.save_dir, splits['scaler'], label_encoders, disease_mapping, feature_names)
    
    # 2. Build or Load Model
    print("\n" + "="*60)
    print("Step 2: Preparing Model")
    print("="*60)
    
    input_dim = splits['X_train'].shape[1]
    model_wrapper = GeneticDiseaseModel(
        input_dim=input_dim, 
        model_size=args.model_size, 
        learning_rate=args.learning_rate, 
        num_classes=num_classes
    )
    
    model_save_path = os.path.join(args.save_dir, 'finetuned_model.keras' if args.finetune else 'kaggle_model.keras')
    
    if args.finetune:
        if os.path.exists(args.load_model):
            print(f"Loading pre-trained model for fine-tuning: {args.load_model}")
            # Use a slightly lower learning rate for more stable fine-tuning
            model_wrapper.learning_rate = 0.0003
            model_wrapper.load(args.load_model)
        else:
            print(f"ERROR: Base model not found at {args.load_model}. Run base training first.")
            sys.exit(1)
    
    model_wrapper.summary()
    
    # 3. Train
    print("\n" + "="*60)
    print(f"Step 3: Training ({'Fine-tuning' if args.finetune else 'Base'})")
    print("="*60)
    
    # Use different patience and batch size for fine-tuning vs base
    patience = 15 if args.finetune else 10
    batch_size = 16 if args.finetune else args.batch_size # Balanced batch for fine-tuning
    
    # Make the scheduler more sensitive for fine-tuning (patience=2)
    sched_patience = 2 if args.finetune else 3
    callbacks = get_callbacks(model_path=model_save_path, patience=patience)
    # Re-configure the scheduler in the callbacks list if needed
    for i, cb in enumerate(callbacks):
        if 'ReduceLROnPlateau' in str(type(cb)):
            cb.patience = sched_patience
    
    history = model_wrapper.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        epochs=10 if args.finetune else args.epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save history
    history_save_path = os.path.join(args.save_dir, 'finetune_history.json' if args.finetune else 'base_history.json')
    save_history_json(history, history_save_path)
    
    # 4. Generate Visualizations
    base_history = None
    if args.finetune:
        base_history = load_history_json(os.path.join(args.save_dir, 'base_history.json'))
    
    plot_history(history, is_finetune=args.finetune, base_history=base_history)
    
    # 5. Evaluation
    print("\n" + "="*60)
    print("Step 4: Evaluating on Test Set")
    print("="*60)
    
    if os.path.exists(model_save_path):
        model_wrapper.load(model_save_path)
        
    y_pred_probs = model_wrapper.model.predict(splits['X_test'])
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(splits['y_test'], y_pred_classes)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    plot_roc_curves(splits['y_test'], y_pred_probs, disease_mapping, is_finetune=args.finetune)
    
    # Confusion Matrix
    cm = confusion_matrix(splits['y_test'], y_pred_classes)
    plt.figure(figsize=(10, 8))
    target_names = [disease_mapping[i] for i in range(num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if not args.finetune else 'Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix ({"Fine-tuned" if args.finetune else "Base"})')
    
    cm_path = f'visualizations/{"finetuned" if args.finetune else "base"}_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    
    print("\n" + "="*60)
    print(f"{'FINE-TUNING' if args.finetune else 'BASE TRAINING'} COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
