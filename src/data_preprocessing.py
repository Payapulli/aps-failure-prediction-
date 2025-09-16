"""
Data preprocessing utilities for APS Failure Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_data(train_path, test_path):
    """
    Load and preprocess the APS failure dataset.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
    
    Returns:
        tuple: (train_features, test_features, train_labels, test_labels)
    """
    # Load data
    train_data = pd.read_csv(train_path, skiprows=20, na_values='na')
    test_data = pd.read_csv(test_path, skiprows=20, na_values='na')
    
    # Separate features and labels
    train_features = train_data.iloc[:, 1:]  # Skip class column
    test_features = test_data.iloc[:, 1:]
    train_labels = train_data['class']
    test_labels = test_data['class']
    
    return train_features, test_features, train_labels, test_labels


def impute_missing_values(train_features, test_features, strategy='mean'):
    """
    Impute missing values using specified strategy.
    
    Args:
        train_features (pd.DataFrame): Training features
        test_features (pd.DataFrame): Test features
        strategy (str): Imputation strategy ('mean', 'median', 'mode')
    
    Returns:
        tuple: (imputed_train_features, imputed_test_features, imputer)
    """
    imputer = SimpleImputer(strategy=strategy)
    
    # Fit on training data only to prevent data leakage
    imputer.fit(train_features)
    
    # Transform both datasets
    train_imputed = imputer.transform(train_features)
    test_imputed = imputer.transform(test_features)
    
    # Convert back to DataFrames
    train_imputed = pd.DataFrame(train_imputed, columns=train_features.columns)
    test_imputed = pd.DataFrame(test_imputed, columns=test_features.columns)
    
    return train_imputed, test_imputed, imputer


def calculate_coefficient_of_variation(features):
    """
    Calculate coefficient of variation for each feature.
    
    Args:
        features (pd.DataFrame): Feature matrix
    
    Returns:
        pd.DataFrame: CV values for each feature
    """
    stats = features.describe()
    cv = stats.loc['std'] / stats.loc['mean']
    return pd.DataFrame({'CV': cv}).sort_values('CV', ascending=False)


def get_class_distribution(labels):
    """
    Get class distribution statistics.
    
    Args:
        labels (pd.Series): Class labels
    
    Returns:
        dict: Class distribution statistics
    """
    counts = labels.value_counts()
    total = len(labels)
    
    return {
        'counts': counts,
        'proportions': counts / total,
        'imbalance_ratio': counts.max() / counts.min()
    }
