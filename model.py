from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np

def train_test_split_chronological(X, y, test_size=0.2):
    """
    Splits data chronologically.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size (float): Proportion of data to use for testing (from the end).
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Calculate split point
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def tune_and_train_model(X_train, y_train, model_type='xgboost'):
    """
    Tunes hyperparameters using Walk-Forward Validation and trains the best model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): 'xgboost' (default)
        
    Returns:
        Trained model with best parameters
    """
    print(f"Tuning and Training {model_type} model on {len(X_train)} samples...")
    
    # TimeSeriesSplit for Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'xgboost':
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Hyperparameter Grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError("Unsupported model type. Choose 'xgboost' or 'random_forest'.")
    
    # Randomized Search
    print("Running RandomizedSearchCV...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring='precision', # Focus on precision (minimizing false positives) for trading
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best Validation Score: {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    
    return best_model

def train_model(X_train, y_train, model_type='xgboost'):
    """Trains without tuning for speed."""
    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model
