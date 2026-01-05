"""
Kaggle Dataset Loader Module

Handles downloading and loading the Genetic Disease Prediction dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/syeddanish5/genetic-disease-prediction-dataset

Features:
- Age, Gender, Family History
- Hemoglobin Level, Fetal Hemoglobin
- RDW_CV, Serum Ferritin
- BRCA1 Expression, p53 Mutation
- And more biomedical features

Target: Genetic diseases (Thalassemia, Hemophilia, Breast Cancer, Sickle Cell Anemia, Cystic Fibrosis)
"""

import os
import sys
import shutil
import pickle
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def download_kaggle_dataset(dataset_name='syeddanish5/genetic-disease-prediction-dataset',
                            output_dir='data'):
    """
    Download dataset from Kaggle using kaggle CLI.
    
    Parameters:
    -----------
    dataset_name : str
        Kaggle dataset identifier (username/dataset-name)
    output_dir : str
        Directory to save the downloaded files
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if 'kaggle' is in PATH
        kaggle_cmd = "kaggle"
        if shutil.which(kaggle_cmd) is None:
            # Fallback to common user bin locations
            possible_paths = [
                os.path.expanduser("~/Library/Python/3.9/bin/kaggle"),
                os.path.expanduser("~/.local/bin/kaggle"),
                os.path.join(sys.prefix, "bin", "kaggle")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    kaggle_cmd = path
                    print(f"Found kaggle at: {kaggle_cmd}")
                    break
        
        # Construct command
        cmd = [
            kaggle_cmd, "datasets", "download", 
            "-d", dataset_name, 
            "-p", output_dir, 
            "--unzip"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Dataset downloaded to {output_dir}")
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing kaggle command: {e}")
        print("Ensure you have set up ~/.kaggle/kaggle.json with your API credentials.")
        return False
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def load_kaggle_dataset(data_path='data/genetic_disease.csv', 
                        target_column='Disease'):
    """
    Load and preprocess the Kaggle Genetic Disease Prediction dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    target_column : str
        Name of the target column
    
    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Encoded labels
    feature_names : list
        List of feature names
    disease_mapping : dict
        Mapping of encoded values to disease names
    """
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    if target_column not in df.columns:
        # Try to find target column
        possible_targets = ['Disease', 'disease', 'Target', 'target', 'Label', 'label']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
    
    y_raw = df[target_column]
    X_df = df.drop(columns=[target_column])

    # Map numeric labels to disease names if using the known Kaggle dataset
    if pd.api.types.is_numeric_dtype(y_raw) and set(y_raw.unique()).issubset({0, 1, 2, 3, 4}):
        print("Detected numeric labels. Applying inferred mapping based on biomedical features.")
        # Mapping inferred from feature analysis:
        # 0: Thalassemia (High Fetal Hb, Low Hb)
        # 1: Hemophilia (Normal-ish Hb, distinct profile)
        # 2: Breast Cancer (High BRCA1/p53)
        # 3: Sickle Cell Anemia (High Sickled RBC, High Fetal Hb)
        # 4: Cystic Fibrosis (High Sweat Chloride)
        disease_name_map = {
            0: 'Thalassemia',
            1: 'Hemophilia',
            2: 'Breast Cancer',
            3: 'Sickle Cell Anemia',
            4: 'Cystic Fibrosis'
        }
        y_raw = y_raw.map(disease_name_map)
    
    # Handle categorical features
    categorical_cols = X_df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le
    
    # Handle missing values
    X_df = X_df.fillna(X_df.median())
    
    # Convert to numpy
    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()
    
    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    disease_mapping = dict(zip(range(len(target_encoder.classes_)), target_encoder.classes_))
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Target classes: {disease_mapping}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names, disease_mapping, label_encoders


def load_or_download_dataset(data_dir='data', force_download=False):
    """
    Load dataset from local path or download from Kaggle if not present.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    force_download : bool
        If True, download even if files exist
    
    Returns:
    --------
    X, y, feature_names, disease_mapping, label_encoders
    """
    # Look for CSV files in data directory
    possible_files = [
        os.path.join(data_dir, 'genetic_disease.csv'),
        os.path.join(data_dir, 'Genetic Disease Prediction.csv'),
        os.path.join(data_dir, 'data.csv'),
        os.path.join(data_dir, 'train.csv'),
    ]
    
    # Also search for any CSV file
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for f in csv_files:
            possible_files.append(os.path.join(data_dir, f))
    
    data_file = None
    for path in possible_files:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None or force_download:
        print("Dataset not found locally. Attempting to download from Kaggle...")
        if download_kaggle_dataset(output_dir=data_dir):
            # Search again for downloaded files
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                data_file = os.path.join(data_dir, csv_files[0])
    
    if data_file is None:
        raise FileNotFoundError(
            f"No dataset found in {data_dir}. Please:\n"
            f"1. Download the dataset from Kaggle:\n"
            f"   https://www.kaggle.com/datasets/syeddanish5/genetic-disease-prediction-dataset\n"
            f"2. Extract the CSV file to the '{data_dir}' folder\n"
            f"\nOr install kaggle CLI and configure API credentials:\n"
            f"   pip install kaggle\n"
            f"   # Set up ~/.kaggle/kaggle.json with your API key"
        )
    
    print(f"Loading dataset from: {data_file}")
    return load_kaggle_dataset(data_file)


def prepare_data_for_multiclass(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare data splits for multi-class classification.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    random_state : int
        Random seed
    
    Returns:
    --------
    dict : Dictionary containing splits and scaler
    """
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler
    }





def save_preprocessing_objects(output_dir, scaler, label_encoders, disease_mapping, feature_names):
    """
    Save preprocessing objects to disk for later use in prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    objects = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'disease_mapping': disease_mapping,
        'feature_names': feature_names
    }
    
    file_path = os.path.join(output_dir, 'preprocessing.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(objects, f)
    
    print(f"Preprocessing objects saved to {file_path}")


def load_preprocessing_objects(input_dir):
    """
    Load preprocessing objects from disk.
    """
    file_path = os.path.join(input_dir, 'preprocessing.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessing objects not found at {file_path}")
        
    with open(file_path, 'rb') as f:
        objects = pickle.load(f)
        
    return objects


if __name__ == "__main__":
    # Test the data loader
    print("="*60)
    print("Testing Kaggle Dataset Loader")
    print("="*60)
    
    # Try to load dataset, download if not found
    try:
        X, y, feature_names, disease_mapping, label_encoders = load_or_download_dataset()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nERROR: Dataset not found. Please setup Kaggle API.")
        exit(1)
    
    print(f"\nData loaded successfully!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {feature_names}")
    print(f"Disease mapping: {disease_mapping}")
    
    # Prepare splits
    splits = prepare_data_for_multiclass(X, y)
    print(f"\nTrain: {splits['X_train'].shape}")
    print(f"Val: {splits['X_val'].shape}")
    print(f"Test: {splits['X_test'].shape}")
