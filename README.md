---
title: Genetic Disease Prediction
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.10.0
python_version: "3.11"
---

# Genetic Disease Risk Prediction

This project is a machine learning-based tool designed to assess the risk of five common genetic disorders: **Thalassemia**, **Hemophilia**, **Sickle Cell Anemia**, **Breast Cancer**, and **Cystic Fibrosis**. It also includes a "Healthy / Low Risk" category for cases where the model's confidence in all known diseases is low.

## 1. Project Overview

The core of this project is a **Deep Forward Neural Network (DFNN)** designed for genomic pattern recognition. It features:
- **High-capacity Model**: Built with Keras/TensorFlow, optimized for multi-class classification.
- **Preprocessing Pipeline**: Robust data cleaning, feature scaling, and mapping of categorical variables using scikit-learn.
- **Gradio Web Interface**: An interactive UI (`app.py`) for making real-time predictions.
- **Feature Selection**: Algorithms to identify the most significant genomic markers (e.g., BRCA1, p53).

## 2. Model Architecture & Parameters

### 🏗️ Network Layers
The model consists of a multi-layer stack:
- **Input layer**: Preprocessed medical and genetic markers.
- **Hidden Layer 1**: 512 neurons (ReLU)
- **Hidden Layer 2**: 256 neurons (ReLU)
- **Hidden Layer 3**: 128 neurons (ReLU)
- **Hidden Layer 4**: 64 neurons (ReLU)
- **Output layer**: 5 neurons (Softmax) for disease risk probabilities.

### ⚙️ Parameters & Optimization
- **Activation Functions**: ReLU in hidden layers for efficient learning; Softmax in the final layer for multi-class probability scores.
- **Overfitting Prevention**:
    - **Dropout**: Rates of [0.5, 0.4, 0.3, 0.2] are applied to disable random neurons during training.
    - **L2 Regularization**: A penalty (0.001) is added to large weights for better generalization.
    - **Normalization**: Batch Normalization after each dense layer to stabilize and speed up learning.
- **Optimizer**: Adam (learning_rate=0.001), an adaptive learning rate algorithm.

## 3. Usage & Commands

### Local Web App (Recommended)
To launch the user-friendly Gradio interface:
```bash
python3 app.py
```
This starts a local server at [http://127.0.0.1:7860](http://127.0.0.1:7860).

### Training the Base Model
To train the model from scratch (default settings: 7 epochs):
```bash
python3 train.py --epochs 7
```

### Fine-tuning the Model
To further improve performance (usually reaching up to 96%+ accuracy):
```bash
python3 train.py --finetune --epochs 7
```

### Model Comparison
To evaluate performance against baseline models (Logistic Regression, Random Forest, Gradient Boosting):
```bash
python3 compare_models.py
```

## 4. Deployment to Hugging Face Spaces

1. **Create a New Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces), click **Create new Space**, choose a name, and select **Gradio** as the SDK.
2. **Select Hardware**: Choose the free CPU tier.
3. **Upload Files**: Upload `app.py`, `requirements.txt`, and the `models/` directory (containing `.keras` model and `preprocessing.pkl`).
4. **Automatic Build**: Hugging Face will automatically install dependencies and launch the app.

> [!IMPORTANT]
> By default, Hugging Face Spaces are **Public**. Anyone with the link can access your tool. You can change this to **Private** under the Settings tab.

## 5. Technical Details
- **Input Features**: Age, Gender, Hemoglobin Levels, BRCA1/p53 Expression, Sweat Chloride, etc.
- **Dataset**: [Genetic Disease Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/syeddanish5/genetic-disease-prediction-dataset).
- **Accuracy**: Achieving up to **96.00%** on validation data after fine-tuning.

---
*Final Year Project - 2026*
