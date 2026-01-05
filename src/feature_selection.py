"""
Feature Selection Module

Implements multiple feature selection methods for high-dimensional genomic data.
"""

import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    mutual_info_classif,
    RFE
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    """
    A comprehensive feature selection class implementing multiple methods
    for genomic data analysis.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the FeatureSelector.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        self.selector = None
        self.method_used = None
    
    def variance_threshold(self, X, threshold=0.01):
        """
        Remove features with low variance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        threshold : float
            Minimum variance threshold
        
        Returns:
        --------
        X_selected : np.ndarray
            Transformed feature matrix
        """
        self.selector = VarianceThreshold(threshold=threshold)
        X_selected = self.selector.fit_transform(X)
        self.selected_features = self.selector.get_support(indices=True)
        self.method_used = 'variance_threshold'
        return X_selected
    
    def chi_square(self, X, y, k=100):
        """
        Select features using chi-square test.
        Note: Features must be non-negative.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        k : int
            Number of top features to select
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        # Ensure non-negative values for chi-square
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
        X_selected = self.selector.fit_transform(X_scaled, y)
        self.selected_features = self.selector.get_support(indices=True)
        self.feature_scores = self.selector.scores_
        self.method_used = 'chi_square'
        return X_selected
    
    def mutual_information(self, X, y, k=100):
        """
        Select features using mutual information.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        k : int
            Number of top features to select
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        self.selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=self.random_state),
            k=min(k, X.shape[1])
        )
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support(indices=True)
        self.feature_scores = self.selector.scores_
        self.method_used = 'mutual_information'
        return X_selected
    
    def recursive_feature_elimination(self, X, y, k=100, step=10):
        """
        Recursive Feature Elimination using Logistic Regression.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        k : int
            Number of features to select
        step : int
            Number of features to remove at each iteration
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        estimator = LogisticRegression(
            max_iter=1000, 
            random_state=self.random_state,
            solver='saga',
            penalty='l2'
        )
        self.selector = RFE(
            estimator=estimator, 
            n_features_to_select=min(k, X.shape[1]),
            step=step
        )
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support(indices=True)
        self.feature_scores = self.selector.ranking_
        self.method_used = 'rfe'
        return X_selected
    
    def lasso_selection(self, X, y, alpha=0.01, k=100):
        """
        Feature selection using LASSO (L1) regularization.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        alpha : float
            Regularization strength
        k : int
            Maximum number of features to select
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=5000)
        lasso.fit(X, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(lasso.coef_)
        
        # Select top k features with non-zero coefficients
        non_zero_mask = importance > 1e-10
        if np.sum(non_zero_mask) == 0:
            # If all coefficients are zero, select top k by importance
            top_indices = np.argsort(importance)[-k:]
        else:
            # Select non-zero features, up to k
            non_zero_indices = np.where(non_zero_mask)[0]
            if len(non_zero_indices) > k:
                top_k_importance = np.argsort(importance[non_zero_indices])[-k:]
                top_indices = non_zero_indices[top_k_importance]
            else:
                top_indices = non_zero_indices
        
        self.selected_features = np.sort(top_indices)
        self.feature_scores = importance
        self.method_used = 'lasso'
        return X[:, self.selected_features]
    
    def random_forest_importance(self, X, y, k=100, n_estimators=100):
        """
        Feature selection using Random Forest feature importance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        k : int
            Number of top features to select
        n_estimators : int
            Number of trees in the forest
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Select top k features
        top_indices = np.argsort(importance)[-min(k, X.shape[1]):]
        self.selected_features = np.sort(top_indices)
        self.feature_scores = importance
        self.method_used = 'random_forest'
        return X[:, self.selected_features]
    
    def select_features(self, X, y, method='mutual_info', k=100, **kwargs):
        """
        Main method to select features using specified method.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        method : str
            Feature selection method: 'variance', 'chi2', 'mutual_info', 
            'rfe', 'lasso', 'random_forest'
        k : int
            Number of features to select
        **kwargs : dict
            Additional arguments for specific methods
        
        Returns:
        --------
        X_selected : np.ndarray
            Selected features
        """
        method_map = {
            'variance': lambda: self.variance_threshold(X, kwargs.get('threshold', 0.01)),
            'chi2': lambda: self.chi_square(X, y, k),
            'mutual_info': lambda: self.mutual_information(X, y, k),
            'rfe': lambda: self.recursive_feature_elimination(X, y, k, kwargs.get('step', 10)),
            'lasso': lambda: self.lasso_selection(X, y, kwargs.get('alpha', 0.01), k),
            'random_forest': lambda: self.random_forest_importance(X, y, k, kwargs.get('n_estimators', 100))
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(method_map.keys())}")
        
        return method_map[method]()
    
    def get_selected_feature_indices(self):
        """Return indices of selected features."""
        return self.selected_features
    
    def get_feature_scores(self):
        """Return feature importance scores."""
        return self.feature_scores
    
    def transform(self, X):
        """Transform new data using previously selected features."""
        if self.selected_features is None:
            raise ValueError("No features selected yet. Run select_features first.")
        return X[:, self.selected_features]
    
    def get_summary(self):
        """Get a summary of feature selection."""
        return {
            'method': self.method_used,
            'n_selected': len(self.selected_features) if self.selected_features is not None else 0,
            'selected_indices': self.selected_features
        }


def compare_feature_selection_methods(X, y, k=100, random_state=42):
    """
    Compare all feature selection methods and return results.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    k : int
        Number of features to select
    random_state : int
        Random seed
    
    Returns:
    --------
    results : dict
        Dictionary with method names as keys and selected feature indices as values
    """
    methods = ['variance', 'chi2', 'mutual_info', 'rfe', 'lasso', 'random_forest']
    results = {}
    
    fs = FeatureSelector(random_state=random_state)
    
    for method in methods:
        print(f"Running {method}...")
        try:
            X_selected = fs.select_features(X, y, method=method, k=k)
            results[method] = {
                'selected_features': fs.get_selected_feature_indices(),
                'n_selected': len(fs.get_selected_feature_indices()),
                'shape': X_selected.shape
            }
        except Exception as e:
            print(f"  Error with {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test feature selection
    from data_generator import generate_genomic_data
    
    print("Generating test data...")
    X, y, feature_names, informative_idx = generate_genomic_data(
        n_samples=500, 
        n_features=200, 
        n_informative=30
    )
    
    print(f"Original data shape: {X.shape}")
    print(f"Truly informative features: {len(informative_idx)}")
    
    # Compare methods
    print("\nComparing feature selection methods...")
    results = compare_feature_selection_methods(X, y, k=50)
    
    print("\n" + "="*50)
    print("Results Summary:")
    print("="*50)
    
    for method, result in results.items():
        if 'error' not in result:
            # Check overlap with truly informative features
            overlap = len(set(result['selected_features']) & set(informative_idx))
            print(f"\n{method}:")
            print(f"  Selected: {result['n_selected']} features")
            print(f"  Overlap with informative: {overlap}/{len(informative_idx)}")
