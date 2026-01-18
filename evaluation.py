from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        y_pred: Predictions
    """
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def simulate_equity_curve(y_pred, prices_test, initial_capital=10000):
    """
    Simulates a trading strategy based on predictions.
    
    Args:
        y_pred (np.array): Predictions (1 for Up/Long, 0 for Ignore/Cash)
        prices_test (pd.Series): Adjusted Close prices corresponding to ONE DAY BEFORE prediction event?
                                 NO. prices_test[i] is the price at time T.
                                 Target[i] was "Return at T+1 > 0" (if Index prediction).
                                 So if Prediction[i] == 1, we Buy at Close[i] and Sell at Close[i+1].
        initial_capital (float): Starting capital.
        
    Returns:
        None (prints result)
    """
    
    # Align predictions with returns
    # We used Features(T) to predict Direction(T+1).
    # Return(T+1) = (Price(T+1) - Price(T)) / Price(T)
    
    # Calculate returns of the underlying asset
    # prices_test aligns with X_test.
    # We need the FUTURE returns that the target was built on.
    
    # Since X_test and y_test were aligned such that y_test[i] describes the return relative to features X_test[i],
    # and we know X_test[i] comes from row i of the original DF...
    
    # We need to compute the returns of the test period.
    # Returns = prices_test.pct_change().shift(-1)
    
    # Note on alignment:
    # prices_test is the Close price series corresponding to X_test index.
    # The Target y_test[i] corresponds to the return from prices_test[i] to prices_test[i+1].
    
    # Let's verify lengths.
    # If X_test has N rows (indices 0 to N-1).
    # prices_test has N rows.
    # We can trade N-1 times (because we need T+1 for the last trade).
    # Ideally our data loader/feature prep dropped the last row where T+1 is unknown.
    # So y_test contains valid targets.
    # This means for every row in X_test, we HAVE a realized target/return.
    # Wait, in features.py/create_targets/prepare_features:
    # We shifted Return(-1) to create Target.
    # Then `dropna()`.
    # So for every row remaining, the Future Return exists.
    
    # We need to re-calculate that actual return to compute equity,
    # or pass it from features (which we dropped).
    # Simplest: Re-calculate from prices.
    
    # We need Future Return for each row i: (Price[i+1] - Price[i]) / Price[i]
    # But Price[i+1] might be in the next batch or computed from this batch?
    # Because of `dropna`, X_test index is contiguous.
    # We can just take prices_test, calculate daily returns, shift(-1).
    
    # BUT, prices_test comes from the main DF index.
    # Let's assume prices_test is the FULL series or we need to look up next day.
    # Actually, simpler:
    # Strategy Return = Prediction * Actual_Next_Day_Return
    
    # We verify if we can compute Actual_Next_Day_Return from prices_test.
    # prices_test indices: t_start, ..., t_end.
    # We need price at t_end + 1 to compute return for the last point?
    # No, the last point in X_test has a valid target y_test.
    # This implies the data for t+1 WAS available when creating y_test before split.
    # But prices_test passed here is likely just X_test['Close'] or aligned series.
    # Use pct_change().shift(-1) on the FULL dataset is best, but here we might only have the slice.
    
    # Approximation/Fix:
    # We can just use the provided prices_test and compute pct_change()
    # But that gives Return(t) = (Price(t) - Price(t-1)) / Price(t-1).
    # We predicted Direction(t+1).
    # So we match Prediction(t) with Return(t+1).
    # Return(t+1) corresponds to index t+1.
    
    returns = prices_test.pct_change().shift(-1)
    # The last value will be NaN unless we look ahead one more step.
    # Since `dropna` in feature engineering removed the last entry where future return wasn't known,
    # `prices_test` here SHOULD conceptually allow us to know the return, 
    # BUT `pct_change().shift(-1)` on this slice `prices_test` will result in the last element being NaN 
    # (because pandas doesn't know the NEXT value outside the series).
    
    # However, since `y_test` is valid, we know that return exists.
    # It's safer to rely on the fact that for classification metrics, we just count wins.
    # But for Equity Curve, we need the MAGNITUDE of returns.
    
    # HACK for simulation:
    # We drop the last prediction for the equity curve calc to avoid the NaN issue if we don't have the next price.
    
    valid_returns = returns.iloc[:-1]
    valid_preds = y_pred[:-1]
    
    # Strategy
    # If Pred=1, we get the Return. If Pred=0, we get 0 (Cash).
    strategy_returns = valid_returns * valid_preds
    
    # Buy & Hold
    # Always own the asset
    bnh_returns = valid_returns
    
    # Cumulative
    equity_strategy = (1 + strategy_returns).cumprod()
    equity_bnh = (1 + bnh_returns).cumprod()
    
    final_strategy = equity_strategy.iloc[-1] * initial_capital
    final_bnh = equity_bnh.iloc[-1] * initial_capital
    
    # --- Advanced Metrics ---
    def calculate_metrics(returns_series):
        # Annualized Sharpe (assuming 0 risk-free for simplicity or excess return)
        sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        
        # Max Drawdown
        cum_ret = (1 + returns_series).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return sharpe, max_dd

    sharpe_strat, dd_strat = calculate_metrics(strategy_returns)
    sharpe_bnh, dd_bnh = calculate_metrics(bnh_returns)
    
    print(f"\n--- Equity Curve Simulation (Initial: ${initial_capital}) ---")
    print(f"Strategy Final: ${final_strategy:,.2f} ({((final_strategy/initial_capital)-1)*100:.2f}%)")
    print(f"  > Sharpe Ratio: {sharpe_strat:.2f}")
    print(f"  > Max Drawdown: {dd_strat*100:.2f}%")
    
    print(f"Buy & Hold Final: ${final_bnh:,.2f} ({((final_bnh/initial_capital)-1)*100:.2f}%)")
    print(f"  > Sharpe Ratio: {sharpe_bnh:.2f}")
    print(f"  > Max Drawdown: {dd_bnh*100:.2f}%")
    
    return equity_strategy, equity_bnh
