"""
Model training utilities for APS Failure Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from xgboost import XGBClassifier


def train_random_forest(X_train, y_train, X_test, y_test, **kwargs):
    """
    Train Random Forest classifier.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        **kwargs: Additional parameters for RandomForestClassifier
    
    Returns:
        dict: Model and performance metrics
    """
    # Default parameters
    params = {
        'n_estimators': 100,
        'random_state': 42,
        'oob_score': True,
        **kwargs
    }
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    oob_error = 1 - model.oob_score_
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'oob_error': oob_error,
        'train_predictions': train_pred,
        'test_predictions': test_pred
    }


def train_xgboost(X_train, y_train, X_test, y_test, use_smote=False, **kwargs):
    """
    Train XGBoost classifier with optional SMOTE.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        use_smote (bool): Whether to use SMOTE for class balancing
        **kwargs: Additional parameters for XGBClassifier
    
    Returns:
        dict: Model and performance metrics
    """
    # Convert labels to binary
    class_mapping = {'neg': 0, 'pos': 1}
    y_train_bin = y_train.map(class_mapping)
    y_test_bin = y_test.map(class_mapping)
    
    # Default parameters
    params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'random_state': 42,
        **kwargs
    }
    
    # Create model
    xgb_model = XGBClassifier(**params)
    
    if use_smote:
        # Use SMOTE pipeline
        smote = SMOTE(random_state=42)
        pipeline = make_imb_pipeline(smote, xgb_model)
        
        # Hyperparameter grid
        param_grid = {
            'xgbclassifier__reg_alpha': [10**i for i in range(-5, 5)]
        }
    else:
        # Direct XGBoost
        pipeline = xgb_model
        param_grid = {
            'reg_alpha': [10**i for i in range(-5, 5)]
        }
    
    # Grid search with cross-validation
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_splitter,
        verbose=0
    )
    
    # Train model
    grid_search.fit(X_train, y_train_bin)
    
    # Make predictions
    train_pred = grid_search.predict(X_train)
    test_pred = grid_search.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train_bin, train_pred)
    test_accuracy = accuracy_score(y_test_bin, test_pred)
    cv_error = 1 - grid_search.best_score_
    
    return {
        'model': grid_search.best_estimator_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_error': cv_error,
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'best_params': grid_search.best_params_
    }


def evaluate_model_performance(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
    
    Returns:
        dict: Performance metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics
