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

from sklearn.ensemble import VotingClassifier

def tune_and_train_model(X_train, y_train, model_type='ensemble', verbose=True):
    """
    Trains a Deep Ensemble Voting Classifier (XGB + RF + GB).
    Step 1: Tune XGBoost for best hyperparameters.
    Step 2: Combine with Random Forest and Gradient Boosting.
    Step 3: Train Voting Classifier (Soft Voting).
    """
    if verbose:
        print(f"Training Deep Ensemble AI on {len(X_train)} samples...")
    
    # TimeSeriesSplit for Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- Component 1: XGBoost (Tuned) ---
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Quick Tuning Grid
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    if verbose: print(">> Tuning XGBoost Component...")
    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=xgb_grid,
        n_iter=15, # Fast but effective
        scoring='precision', # High precision preferred for trading
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    
    if verbose: print(f"   Best XGB Params: {search.best_params_}")
    
    # --- Component 2: Random Forest (Bagging) ---
    # Robust default parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # --- Component 3: Gradient Boosting (sklearn) ---
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    # --- Ensemble: Voting Classifier ---
    if verbose: print(">> Constructing Deep Voting Ensemble...")
    
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft', # Probabilities are averaged
        weights=[2, 1, 1] # Weight XGB higher as it's tuned
    )
    
    # Train the Ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble

def train_model(X_train, y_train, model_type='xgboost'):
    """Deprecated: Use tune_and_train_model for ensemble."""
    return tune_and_train_model(X_train, y_train, verbose=False)
