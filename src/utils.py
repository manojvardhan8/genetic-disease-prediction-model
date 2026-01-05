"""
Utility Functions Module

Helper functions for the genetic disease prediction project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_gpu():
    """Configure GPU settings for TensorFlow."""
    try:
        import tensorflow as tf
        
        # Allow memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"GPU setup error: {e}")


def create_experiment_directory(base_dir='experiments'):
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config(config, filepath):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {filepath}")


def load_config(filepath):
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_feature_importance(feature_scores, feature_names=None, top_n=20, 
                           title='Feature Importance', save_path=None):
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    feature_scores : np.ndarray
        Feature importance scores
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_scores))]
    
    # Get top features
    top_indices = np.argsort(feature_scores)[-top_n:]
    top_scores = feature_scores[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_scores)))
    bars = ax.barh(range(len(top_scores)), top_scores, color=colors)
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_class_distribution(y, title='Class Distribution', save_path=None):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    classes, counts = np.unique(y, return_counts=True)
    colors = ['#3498db', '#e74c3c']
    labels = ['Healthy', 'Disease']
    
    axes[0].bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    for i, (count, label) in enumerate(zip(counts, labels)):
        axes[0].text(i, count + max(counts) * 0.02, str(count), 
                    ha='center', fontsize=12, fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0.05),
               textprops={'fontsize': 12})
    axes[1].set_title('Class Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_correlation_matrix(X, feature_names=None, top_n=30, save_path=None):
    """
    Plot correlation matrix for top features.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    feature_names : list
        Names of features
    top_n : int
        Number of features to include
    save_path : str
        Path to save the plot
    """
    if feature_names is None:
        feature_names = [f'F_{i}' for i in range(X.shape[1])]
    
    # Calculate variance and select top features
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-min(top_n, X.shape[1]):]
    
    X_subset = X[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]
    
    # Calculate correlation
    corr = np.corrcoef(X_subset.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
               square=True, linewidths=0.5, annot=False,
               xticklabels=selected_names, yticklabels=selected_names, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}")
    
    plt.show()
    return fig


def print_data_summary(X, y, feature_names=None):
    """Print comprehensive data summary."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nDataset Shape: {X.shape[0]} samples × {X.shape[1]} features")
    
    print(f"\nClass Distribution:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        pct = count / len(y) * 100
        label = 'Disease' if cls == 1 else 'Healthy'
        print(f"  {label} (class {cls}): {count} ({pct:.1f}%)")
    
    print(f"\nFeature Statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std: {np.std(X):.4f}")
    print(f"  Min: {np.min(X):.4f}")
    print(f"  Max: {np.max(X):.4f}")
    
    # Check for missing values
    missing = np.isnan(X).sum()
    print(f"\nMissing Values: {missing}")
    
    if feature_names:
        print(f"\nFeature Names Sample: {feature_names[:5]}...")


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def get_model_summary_string(model):
    """Get model summary as a string."""
    string_list = []
    model.get_model().summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Set seed
    set_random_seed(42)
    
    # Create experiment directory
    exp_dir = create_experiment_directory()
    print(f"Created experiment directory: {exp_dir}")
    
    # Test data summary
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 2, 100)
    print_data_summary(X, y)
    
    print("\nUtility functions test completed!")
