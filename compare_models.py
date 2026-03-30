import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.kaggle_data_loader import load_or_download_dataset, prepare_data_for_multiclass
from src.model import GeneticDiseaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def compare():
    print("Loading data for comparison...")
    X, y, feature_names, disease_mapping, label_encoders = load_or_download_dataset('data')
    num_classes = len(disease_mapping)
    splits = prepare_data_for_multiclass(X, y, random_state=42)
    
    results = {}
    
    # 1. Baseline Models
    print("\nEvaluating Baseline Models...")
    models_bl = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    for name, model in models_bl.items():
        model.fit(splits['X_train'], splits['y_train'])
        acc = accuracy_score(splits['y_test'], model.predict(splits['X_test']))
        results[name] = acc
        print(f"{name:20}: {acc:.4f}")
    
    # 2. Deep Learning Models
    models_dl = {
        'Deep Forward (Base)': 'models/kaggle_model.keras',
        'Deep Forward (Fine-tuned)': 'models/finetuned_model.keras'
    }
    
    for name, path in models_dl.items():
        if os.path.exists(path):
            print(f"\nEvaluating {name} from {path}...")
            model = GeneticDiseaseModel(input_dim=splits['X_train'].shape[1], num_classes=num_classes)
            model.load(path)
            y_pred_probs = model.model.predict(splits['X_test'], verbose=0)
            if num_classes > 2:
                y_pred_classes = np.argmax(y_pred_probs, axis=1)
            else:
                y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
            
            acc = accuracy_score(splits['y_test'], y_pred_classes)
            results[name] = acc
            print(f"{name:25}: {acc:.4f}")
        else:
            print(f"\nModel file not found for {name}: {path}")

    # 3. Summary and Visualization
    print("\n" + "="*50)
    print("FINAL ACCURACY COMPARISON SUMMARY")
    print("="*50)
    
    # Sort results for better display (Baselines first, then DL)
    sorted_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Deep Forward (Base)', 'Deep Forward (Fine-tuned)']
    final_results = {name: results.get(name) for name in sorted_names if name in results}
    
    for name, acc in final_results.items():
        print(f"{name:25}: {acc:.4f}")
        
    if final_results:
        os.makedirs('visualizations', exist_ok=True)
        plt.figure(figsize=(14, 7))
        names = list(final_results.keys())
        values = list(final_results.values())
        
        # Color palette: Baselines in one color, DF Models in another
        colors = ['#87CEEB' if 'Deep' not in n else '#FF6347' for n in names]
        
        sns.barplot(x=values, y=names, palette=colors, hue=names, legend=False)
        plt.title('Accuracy Comparison: Baseline vs Deep Forward Models', fontsize=14)
        plt.xlabel('Accuracy', fontsize=12)
        plt.xlim(0.8, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('visualizations/final_comparison.png')
        print(f"\nComparison chart saved to: visualizations/final_comparison.png")

if __name__ == "__main__":
    compare()
