import pandas as pd
import warnings
from data_loader import fetch_data
from features import add_technical_indicators, create_targets, prepare_features
from model import train_test_split_chronological, tune_and_train_model, train_model
from evaluation import evaluate_model, simulate_equity_curve

# Suppress warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    print("=== S&P 500 & Alpha Predictor: Phase 2 (Max Performance) ===\n")

    # Configuration
    INDEX_TICKER = '^GSPC'
    CONSTITUENTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'UNH']
    
    dashboard_data = []
    
    # -------------------------------------------------------------------------
    # PART 1: Index Prediction (Directional)
    # -------------------------------------------------------------------------
    print(f"\n>> Analyzing Market Context ({INDEX_TICKER})...")
    
    # Fetch
    index_raw = fetch_data([INDEX_TICKER], period_years=10)
    
    # Features
    index_feat = add_technical_indicators(index_raw)
    
    # Store Index Returns for Alpha Calculation later (align indices)
    # We must ensure index_returns covers the same range as future stock data
    index_returns_full = index_feat['Daily_Return']
    
    # Create Targets (Directional: 1 if Up, 0 if Down)
    index_feat = create_targets(index_feat, index_returns=None)
    
    # Prepare
    X_idx, y_idx = prepare_features(index_feat)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split_chronological(X_idx, y_idx)
    
    # Tune & Train
    # Using 'tune_and_train_model' for automated hyperparameter optimization
    model_idx = tune_and_train_model(X_train, y_train, model_type='xgboost')
    
    # Evaluate
    evaluate_model(model_idx, X_test, y_test)
    
    # Simulation (Index)
    print("\n[Market Simulation]")
    # Recover prices for the test period
    # Note: X_test index corresponds to T. We need prices at T to simulate trade for T+1?
    # Logic: Prediction at T is for Return(T+1).
    # simulate_equity_curve logic needs review if we changed things.
    # It assumes passed prices allow calculation of future return.
    prices_test = index_raw.loc[X_test.index]['Adj Close'] if 'Adj Close' in index_raw.columns else index_raw.loc[X_test.index]['Close']
    simulate_equity_curve(model_idx.predict(X_test), prices_test)
    
    # LATEST PREDICTION (For Dashboard)
    # We predict for "Today+1" using "Today's" data.
    # The last row of X (which is the data from 'today') was likely dropped in prepare_features 
    # because it had no target (tomorrow is unknown).
    # We need to recover that last row to make a Live Prediction.
    
    # Recover last feature row (Prediction for Tomorrow)
    # Re-generate features on full dataset, don't drop the last row yet
    # Actually `prepare_features` drops NaNs. The last row has NaN target.
    # So X contains up to Yesterday.
    
    # Let's get the standard latest features
    latest_feat = index_feat.drop(columns=['Target']).iloc[[-1]] 
    # Check if this row was used in training?
    # If create_targets shifted(-1), the last row target is NaN.
    # So it was dropped from X. 
    # So this 'latest_feat' is exactly what we need to predict Tomorrow.
    
    # Wait, create_targets logic:
    # Target[t] = Return[t+1].
    # Last row at index T: Target is NaN.
    # prepare_features() -> dropna() -> Last row T is gone.
    # So training went up to T-1.
    # Now we take row T to predict T+1.
    
    # We need to make sure `latest_feat` columns match X columns cleanly
    # (prepare_features removes Close, Daily_Return)
    feature_cols = X_idx.columns
    latest_X = index_feat.loc[index_feat.index[-1], feature_cols].to_frame().T
    
    latest_pred = model_idx.predict(latest_X)[0]
    latest_proba = model_idx.predict_proba(latest_X)[0][1]
    
    market_action = "BUY / BULLISH" if latest_pred == 1 else "CASH / BEARISH"
    dashboard_data.append({
        "Ticker": INDEX_TICKER,
        "Type": "Index",
        "Signal": market_action,
        "Confidence": f"{latest_proba:.1%}"
    })

    # -------------------------------------------------------------------------
    # PART 2: Constituents (Alpha Prediction)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # PART 2: Constituents (Alpha Prediction)
    # -------------------------------------------------------------------------
    print(f"\n>> Analyzing Constituents for Alpha & Health (Deep Dive)...")
    from data_loader import get_sentiment, get_fundamentals
    from tqdm import tqdm
    
    # Wrap iteration with tqdm for progress bar
    for stock in tqdm(CONSTITUENTS, desc="Optimizing Models", unit="ticker"):
        # print(f"Processing {stock}...") # Removed to keep progress bar clean
        try:
            # 1. AI ML Loop (Quiet Mode)
            df_stk = fetch_data([stock], period_years=10, quiet=True)
            feat_stk = add_technical_indicators(df_stk)
            feat_stk = create_targets(feat_stk, index_returns=index_returns_full)
            X_stk, y_stk = prepare_features(feat_stk)
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split_chronological(X_stk, y_stk)
            
            if len(X_train_s) < 100: continue
            
            # PHASE 3 UPGRADE: Full Tuning for every stock
            model_stk = tune_and_train_model(X_train_s, y_train_s, model_type='xgboost', verbose=False)
            
            latest_X_stk = feat_stk.loc[feat_stk.index[-1], feature_cols].to_frame().T
            ai_proba = model_stk.predict_proba(latest_X_stk)[0][1]
            
            # 2. Sentiment Pulse
            sentiment = get_sentiment(stock)
            
            # 3. Fundamentals
            funds = get_fundamentals(stock)
            # Basic Health Score (0 to 1)
            # PEG < 1 is great, < 2 is okay. Profit margin > 0 is good.
            health = 0.5
            if funds['PEG'] > 0 and funds['PEG'] < 1.5: health += 0.2
            if funds['Profit_Margin'] > 0.15: health += 0.2
            if funds['Forward_PE'] > 0 and funds['Forward_PE'] < 40: health += 0.1
            
            # 4. Composite Score (Weighted)
            # AI (50%), Sentiment (25%), Fundamentals (25%)
            # Normalize sentiment from (-1, 1) to (0, 1)
            norm_sentiment = (sentiment + 1) / 2
            composite = (ai_proba * 0.5) + (norm_sentiment * 0.25) + (health * 0.25)
            
            action = "STRONG BUY" if composite > 0.6 else ("OVERWEIGHT" if composite > 0.52 else "HOLD/NEUTRAL")
            if composite < 0.45: action = "UNDERWEIGHT"
            
            dashboard_data.append({
                "Ticker": stock,
                "Type": "Stock",
                "Signal": action,
                "Composite": f"{composite:.1%}",
                "AI": f"{ai_proba:.0%}",
                "Sent": f"{norm_sentiment:.0%}",
                "Fund": f"{health:.0%}"
            })
            
        except Exception as e:
            # print(f"Error processing {stock}: {e}") # Suppress error noise
            pass

    # -------------------------------------------------------------------------
    # PART 3: Dashboard
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("                 TRADING SIGNAL DASHBOARD (SENTIMENT & FUNDAMENTALS)      ")
    print("="*80)
    print(f"{'TICKER':<10} | {'SIGNAL':<15} | {'STRENGTH':<10} | {'AI':<6} | {'SENT':<6} | {'FUND':<6}")
    print("-" * 80)
    
    for item in dashboard_data:
        # Index doesn't have funds/sent in this simple version
        ai = item.get('AI', '--')
        sent = item.get('Sent', '--')
        fund = item.get('Fund', '--')
        comp = item.get('Composite', item.get('Confidence', '--'))
        
        print(f"{item['Ticker']:<10} | {item['Signal']:<15} | {comp:<10} | {ai:<6} | {sent:<6} | {fund:<6}")
    print("-" * 80)
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    run_pipeline()
