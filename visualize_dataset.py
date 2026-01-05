import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.kaggle_data_loader import load_or_download_dataset

def visualize_dataset(output_dir='visualizations'):
    """
    Generate Exploratory Data Analysis (EDA) plots for the Genetic Disease Dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    X, y, feature_names, disease_mapping, _ = load_or_download_dataset()
    
    # Create DataFrame for better visualization with feature names and labels
    df = pd.DataFrame(X, columns=feature_names)
    # Map y (indices) back to disease names using disease_mapping
    # disease_mapping is {0: 'Thalassemia', ...}
    df['Disease'] = [disease_mapping[i] for i in y]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['Disease'].unique()}")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y='Disease', data=df, order=df['Disease'].value_counts().index, palette='viridis')
    plt.title('Disease Class Distribution', fontsize=16)
    plt.xlabel('Count')
    plt.ylabel('Disease')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    print(f"Saved class_distribution.png")
    plt.close()
    
    # 2. Key Bio-markers Boxplots
    # We select interesting features
    key_features = [
        'Hemoglobin', 'Fetal_Hemoglobin', 'Sweat_Chloride', 
        'Sickled_RBC_Percent', 'Serum_Ferritin', 'BRCA1_Expression'
    ]
    
    # Filter only existing columns just in case
    key_features = [f for f in key_features if f in df.columns]
    
    for feature in key_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Disease', y=feature, data=df, palette='Set2')
        plt.title(f'{feature} Distribution by Disease', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_boxplot.png'))
        print(f"Saved {feature}_boxplot.png")
        plt.close()
        
    # 3. Correlation Heatmap
    # Select numeric columns only
    numeric_df = df.drop(columns=['Disease'])
    if numeric_df.shape[1] > 20:
        # If too many features, select top variance ones or just known biomarkers
        # For readability, we stick to the provided key features + a few others
        # Check actual columns
        cols = numeric_df.columns[:20] # Take first 20 if many
        corr_matrix = numeric_df[cols].corr()
    else:
        corr_matrix = numeric_df.corr()
        
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    print(f"Saved correlation_heatmap.png")
    plt.close()
    
    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == "__main__":
    visualize_dataset()
