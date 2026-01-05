"""
Deep Learning Model Module

Neural network architecture for genetic disease risk prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
import numpy as np


def create_disease_prediction_model(input_dim, 
                                     hidden_layers=[512, 256, 128, 64],
                                     dropout_rates=[0.5, 0.4, 0.3, 0.2],
                                     l2_reg=0.001,
                                     learning_rate=0.001,
                                     num_classes=2):
    """
    Create a deep neural network for disease risk prediction.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_layers : list
        Number of neurons in each hidden layer
    dropout_rates : list
        Dropout rate for each hidden layer
    l2_reg : float
        L2 regularization strength
    learning_rate : float
        Initial learning rate
    num_classes : int
        Number of output classes (2 for binary, >2 for multi-class)
    
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    model = keras.Sequential(name='GeneticDiseasePredictor')
    
    # Input layer with batch normalization
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    
    # Hidden layers
    for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
        model.add(layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Activation('relu', name=f'relu_{i+1}'))
        model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
    
    # Output layer - binary vs multi-class
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 
                   keras.metrics.AUC(name='auc'),
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall')]
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_smaller_model(input_dim, learning_rate=0.001):
    """
    Create a smaller model for quick experiments.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    learning_rate : float
        Initial learning rate
    
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    return create_disease_prediction_model(
        input_dim=input_dim,
        hidden_layers=[128, 64, 32],
        dropout_rates=[0.3, 0.2, 0.1],
        l2_reg=0.01,
        learning_rate=learning_rate
    )


def create_larger_model(input_dim, learning_rate=0.0005):
    """
    Create a larger model for more complex datasets.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    learning_rate : float
        Initial learning rate
    
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    return create_disease_prediction_model(
        input_dim=input_dim,
        hidden_layers=[1024, 512, 256, 128, 64],
        dropout_rates=[0.5, 0.5, 0.4, 0.3, 0.2],
        l2_reg=0.0005,
        learning_rate=learning_rate
    )


def get_callbacks(model_path='models/best_model.keras', 
                  patience=15, 
                  min_delta=0.001,
                  log_dir=None):
    """
    Get training callbacks.
    
    Parameters:
    -----------
    model_path : str
        Path to save the best model
    patience : int
        Number of epochs to wait before early stopping
    min_delta : float
        Minimum change to qualify as an improvement
    log_dir : str
        Directory for TensorBoard logs
    
    Returns:
    --------
    callbacks : list
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            min_delta=min_delta,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    if log_dir:
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    return callbacks


class GeneticDiseaseModel:
    """
    Wrapper class for the genetic disease prediction model.
    """
    
    def __init__(self, input_dim, model_size='default', learning_rate=0.001, num_classes=2):
        """
        Initialize the model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        model_size : str
            'small', 'default', or 'large'
        learning_rate : float
            Initial learning rate
        num_classes : int
            Number of output classes (2 for binary, >2 for multi-class)
        """
        self.input_dim = input_dim
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = self._create_model()
        self.history = None
    
    def _create_model(self):
        """Create the model based on size specification."""
        if self.model_size == 'small':
            return create_disease_prediction_model(
                self.input_dim, 
                hidden_layers=[128, 64, 32],
                dropout_rates=[0.3, 0.2, 0.1],
                l2_reg=0.01,
                learning_rate=self.learning_rate,
                num_classes=self.num_classes
            )
        elif self.model_size == 'large':
            return create_disease_prediction_model(
                self.input_dim,
                hidden_layers=[1024, 512, 256, 128, 64],
                dropout_rates=[0.5, 0.5, 0.4, 0.3, 0.2],
                l2_reg=0.0005,
                learning_rate=self.learning_rate,
                num_classes=self.num_classes
            )
        else:
            return create_disease_prediction_model(
                self.input_dim, 
                learning_rate=self.learning_rate,
                num_classes=self.num_classes
            )
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, callbacks=None):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        callbacks : list
            Training callbacks
        
        Returns:
        --------
        history : keras.callbacks.History
            Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        if callbacks is None:
            callbacks = get_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_classes(self, X, threshold=0.5):
        """Make class predictions."""
        predictions = self.predict(X)
        return (predictions > threshold).astype(int).flatten()
    
    def evaluate(self, X, y):
        """Evaluate the model."""
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, filepath):
        """Save the model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model(self):
        """Return the underlying Keras model."""
        return self.model


if __name__ == "__main__":
    # Test the model
    print("Creating and testing model...")
    
    # Create a test model
    model = GeneticDiseaseModel(input_dim=100, model_size='small')
    model.summary()
    
    # Create dummy data for testing
    X_train = np.random.randn(500, 100)
    y_train = np.random.randint(0, 2, 500)
    X_val = np.random.randn(100, 100)
    y_val = np.random.randint(0, 2, 100)
    
    print("\nTraining model for 5 epochs...")
    history = model.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=5, 
        batch_size=32,
        callbacks=[]  # No callbacks for quick test
    )
    
    print("\nModel test completed successfully!")
