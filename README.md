# Genetic Disease Risk Prediction

A deep learning project for predicting genetic disease risk using feature selection on genomic data.

## Project Structure

```
├── data/           # Dataset files (download from Kaggle)
├── models/         # Saved model checkpoints
├── src/            # Source code modules
│   ├── kaggle_data_loader.py # Kaggle dataset loader
│   ├── feature_selection.py # Feature selection methods
│   ├── model.py             # Deep learning architecture
│   └── utils.py             # Utility functions

├── main_kaggle.py     # Main script (Kaggle dataset)
├── predict.py         # Inference script

├── visualize_dataset.py # EDA visualization script
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt

# For Kaggle download (optional)
pip install kaggle
```

## Dataset

This project uses the **Genetic Disease Prediction Dataset** from Kaggle:
- **Dataset Link**: [Genetic Disease Prediction](https://www.kaggle.com/datasets/syeddanish5/genetic-disease-prediction-dataset)
- **Features**: Age, Gender, Hemoglobin Levels, BRCA1 Expression, etc.
- **Target**: 6 Classes (Healthy, Thalassemia, Hemophilia, etc.)

### Download Dataset
Automatic download via `kaggle` CLI is supported.

**Manual Setup:**
1. Install and configure `kaggle` CLI:
   ```bash
   pip install kaggle
   # Setup ~/.kaggle/kaggle.json
   ```
2. Or use the command manually:
   ```bash
   kaggle datasets download -d syeddanish5/genetic-disease-prediction-dataset -p data/ --unzip
   ```

## Usage

### With Kaggle Dataset (Recommended)
```bash


# Train with downloaded Kaggle dataset
python3 main.py --epochs 50

# With feature selection
python3 main.py --feature-selection mutual_info --num-features 10

### Making Predictions (Inference)
Once the model is trained, use `predict.py` to classify new patients:

```bash
# Interactive mode (enter values manually)
python3 predict.py --interactive

# Quick test with default sample
python3 predict.py
```



### Visualization (EDA)
Generate analysis plots for the dataset:
```bash
python3 visualize_dataset.py
```



## Feature Selection Methods

- Variance Threshold
- Chi-Square Test
- Mutual Information
- Recursive Feature Elimination (RFE)
- LASSO (L1 Regularization)
- Random Forest Feature Importance

## Author

Final Year Project - 2026
