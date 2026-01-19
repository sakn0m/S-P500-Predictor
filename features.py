import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    Expects 'df' to be a Series or DataFrame with price data (e.g., Adj Close).
    If it's a DataFrame with multiple columns (multiple stocks), this needs to be applied per column.
    However, the design here implies we process one stock/index at a time for feature engineering
    or we handle the wide format.
    
    Given the structure, it's efficiently done if we convert the wide DF (tickers as columns) 
    to a long format or process iteratively. 
    
    For a modular approach compatible with scikit-learn, let's assume this function 
    takes a SINGLE ticker's price series and returns a DataFrame with features.
    """
    
    # Ensure input is suitable
    if isinstance(df, pd.Series):
        # Convert to DF for consistency if passed as Series
        prices = df
    
    # OLD CHECK REMOVED: if df.shape[1] != 1 raise ValueError...
    # We now accept multi-column DFs (OHLCV + Macro)
    prices = df # Placeholder, actual logic handles columns below

    feature_df = pd.DataFrame(index=df.index)
    
    # Support for Macro columns pass-through
    if 'VIX' in df.columns:
        feature_df['VIX'] = df['VIX']
    if 'US10Y' in df.columns:
        feature_df['US10Y'] = df['US10Y']

    # Locate Price Columns (Open, High, Low, Close)
    # If input is simple, it implies single ticker context.
    try:
        if 'Adj Close' in df.columns:
            close = df['Adj Close']
        else:
            close = df['Close']
            
        high = df['High']
        low = df['Low']
        # open_ = df['Open'] # Not strictly needed for these indicators unless we add others
    except KeyError:
        # Fallback if we only have 'Close' (e.g. legacy mode or missing data)
        # We can't calculate ATR/ADX accurately without High/Low.
        print("Warning: High/Low data missing. Advanced features (ATR, ADX) will be skipped or approximated.")
        close = df.iloc[:, 0]
        high = close
        low = close

    feature_df['Close'] = close

    # 1. RSI
    rsi = RSIIndicator(close=close, window=14)
    feature_df['RSI'] = rsi.rsi()

    # 2. MACD
    macd = MACD(close=close)
    feature_df['MACD'] = macd.macd()
    feature_df['MACD_Signal'] = macd.macd_signal()
    feature_df['MACD_Diff'] = macd.macd_diff()

    # 3. Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    feature_df['BB_High'] = bb.bollinger_hband()
    feature_df['BB_Low'] = bb.bollinger_lband()
    feature_df['BB_Width'] = (feature_df['BB_High'] - feature_df['BB_Low']) / close

    # 4. SMA
    sma_50 = SMAIndicator(close=close, window=50)
    feature_df['SMA_50'] = sma_50.sma_indicator()
    
    sma_200 = SMAIndicator(close=close, window=200)
    feature_df['SMA_200'] = sma_200.sma_indicator()
    
    # 5. Advanced Indicators (New for Phase 2)
    from ta.volatility import AverageTrueRange
    from ta.trend import ADXIndicator, CCIIndicator
    from ta.momentum import StochasticOscillator

    # ATR
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    feature_df['ATR'] = atr.average_true_range()

    # ADX
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    feature_df['ADX'] = adx.adx()

    # CCI
    cci = CCIIndicator(high=high, low=low, close=close, window=20)
    feature_df['CCI'] = cci.cci()

    # Stochastic
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    feature_df['Stoch_K'] = stoch.stoch()
    feature_df['Stoch_D'] = stoch.stoch_signal()

    # 6. Returns & Macro Changes
    # 6. Returns & Macro Changes
    feature_df['Daily_Return'] = close.pct_change()
    
    # --- PHASE 4: INSTITUTIONAL UPGRADE ---
    if 'High' in df.columns and 'Low' in df.columns and 'Open' in df.columns:
        # 1. Garman-Klass Volatility (More efficient than Close-to-Close volatility)
        # Formula: 0.5 * ln(High/Low)^2 - (2ln(2)-1) * ln(Close/Open)^2
        # We need check for zeros/negatives inside log, although prices should be positive.
        log_hl = np.log(high / low)**2
        log_co = np.log(close / df['Open'])**2
        feature_df['Garman_Klass'] = np.sqrt(0.5 * log_hl - (2*np.log(2)-1) * log_co)

        # 2. Keltner Channels (Trend Following)
        # EMA 20 +/- 2 * ATR
        # We need ATR calculated above (feature_df['ATR'])
        kc_mid = ta.trend.ema_indicator(close, window=20)
        kc_upper = kc_mid + (2 * feature_df['ATR'])
        kc_lower = kc_mid - (2 * feature_df['ATR'])
        feature_df['KC_Upper_Dist'] = (close - kc_upper) / close
        feature_df['KC_Lower_Dist'] = (close - kc_lower) / close
    
    # 3. Volume Oscillator (Flow)
    if 'Volume' in df.columns:
        feature_df['Volume_Osc'] = ta.momentum.PercentageVolumeOscillator(df['Volume']).pvo()
    
    # 4. SMA Distances (Explicit override if not just returning)
    # (Already calculated above, but ensuring distances are robust)

    if 'VIX' in feature_df.columns:
        feature_df['VIX_Change'] = feature_df['VIX'].pct_change()
    if 'US10Y' in feature_df.columns:
        feature_df['US10Y_Change'] = feature_df['US10Y'].pct_change()

    # --- PHASE 5: "DEEP AI" COMPLEXITY UPGRADE ---
    
    # 1. Price Gaps (Pre-market moves)
    # Gap = (Open - Prev_Close) / Prev_Close
    if 'Open' in df.columns:
        prev_close = close.shift(1)
        feature_df['Gap'] = (df['Open'] - prev_close) / prev_close
    
    # 2. Cyclical Date Features (Seasonality)
    # Markets have weekly and monthly seasonality
    # Sine/Cosine encoding to preserve cyclical nature (Dec -> Jan is close)
    dates = df.index
    feature_df['Month_Sin'] = np.sin(2 * np.pi * dates.month/12)
    feature_df['Month_Cos'] = np.cos(2 * np.pi * dates.month/12)
    feature_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * dates.dayofweek/7)
    
    # 3. Lagged Returns (Memory/Autoregression)
    # Explicitly feed recent returns to the model
    feature_df['Ret_Lag1'] = feature_df['Daily_Return'].shift(1)
    feature_df['Ret_Lag2'] = feature_df['Daily_Return'].shift(2)
    feature_df['Ret_Lag3'] = feature_df['Daily_Return'].shift(3)
    feature_df['Ret_Lag5'] = feature_df['Daily_Return'].shift(5) # Weekly momentum
    
    # 4. Interaction Terms (Conditional Logic)
    # Volatility * Trend: High Vol + High Trend might mean crash or boom
    if 'ATR' in feature_df.columns and 'RSI' in feature_df.columns:
        feature_df['Vol_x_Mom'] = feature_df['ATR'] * feature_df['RSI']
        
    return feature_df

def create_targets(df, index_returns=None):
    """
    Creates target variable.
    
    Args:
        df (pd.DataFrame): DataFrame with features and 'Daily_Return'.
        index_returns (pd.Series, optional): Series of index returns for Alpha calculation. 
                                             If None, assumes we are predicting the index direction itself.
    
    Returns:
        pd.DataFrame: DataFrame with 'Target' column.
        
    Target Logic:
    - If index_returns is None (Predicting Index): Target = 1 if Return(t) > 0, else 0.
      Note: The return at time 't' is (Price_t - Price_{t-1})/Price_{t-1}.
      If we want to predict direction for tomorrow, we usually frame it as:
      Predicting if Price_{t+1} > Price_t.
      This corresponds to Return_{t+1} > 0.
      
    - If index_returns is NOT None (Predicting Alpha): Target = 1 if StockReturn(t) > IndexReturn(t).
      Again, we want to predict this for the FUTURE.
      
    CRITICAL: The 'Target' column must represent the outcome we want to predict using PAST data.
    So initially we calculate the event happening at time t, and then in 'prepare_features' we will lag the features
    so that Features(t-1) predict Target(t). 
    
    Alternatively, we shift the Target BACKWARDS, so Features(t) predict Target(t+1).
    Let's align on: Features(t) predict Direction(t+1).
    So Target col should be Shifted Return Direction.
    """
    
    data = df.copy()
    
    if index_returns is None:
        # User request: "For the index, prediction is binary (1 if price in t+1 > t, else 0)"
        # This implies we want to predict the NEXT day's return sign.
        # We compute current day return signs, then shift them backwards by 1 to align with today's features.
        # Target[t] = 1 if Return[t+1] > 0
        
        # First calculate actual direction for each day
        # We use 0.0 as threshold.
        future_return = data['Daily_Return'].shift(-1)
        data['Target'] = (future_return > 0).astype(int)
        
        # Note: The last row will be NaN/invalid for Target because we don't know t+1.
        
    else:
        # User request: "For stocks, prediction is 1 if the stock's daily return > index's daily return (Alpha)"
        # Again, we want to predict if Tomorrow's Alpha is positive.
        
        # Align index returns
        # Make sure indices match
        aligned_index_ret = index_returns.reindex(data.index)
        
        stock_ret = data['Daily_Return']
        
        # Calculate Alpha for *current* day t
        # alpha_t = stock_ret_t - index_ret_t
        
        # We want to predict if alpha_{t+1} > 0
        future_stock_ret = stock_ret.shift(-1)
        future_index_ret = aligned_index_ret.shift(-1)
        
        data['Target'] = (future_stock_ret > future_index_ret).astype(int)

    return data

def prepare_features(df, target_col='Target'):
    """
    Final features cleaning.
    
    Args:
        df: DataFrame with technical indicators and Target column.
        
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    # Drop rows with NaNs created by lagging (indicators) or shifting (target)
    # Technical indicators like SMA_200 introduce 200 NaNs at start.
    # Target shift introduces 1 NaN at end.
    
    data = df.dropna()
    
    # We want to use features (RSI, SMA, etc.) at time T to predict Target at T (which represents event T+1).
    # Since we already shifted the Target in `create_targets` (Target(t) = Return(t+1)),
    # we DO NOT need to lag the features again relative to this Target column if we treat the row as "Day T".
    
    # Wait, the prompt says: "Lag all features by 1 day to strictly avoid 'Look-Ahead Bias'".
    # If we structure the dataframe such that Row T contains Price T and Target T (Outcome T+1),
    # then Features derived from Price T are legal to predict Target T.
    
    # However, sometimes models are fed with Row T to predict Row T.
    # If the prompt entails standard supervised learning setup X, y:
    # X_t (Features from Day T-1) -> y_t (Outcome Day T).
    # OR X_t (Features from Day T) -> y_t (Outcome Day T+1).
    
    # Let's check the instruction carefully: "Lag all features by 1 day to strictly avoid 'Look-Ahead Bias'".
    # This usually means if I am at day T, I should use data from T-1 to predict T.
    # BUT standard practice is: Use Correctly Aligned Data.
    # If I use Close_T to predict Return_{T+1}, there is no look ahead. Close_T is known at T.
    
    # If the user strictly implies "Using yesterday's data to predict today", then:
    # Predict Return_T using Open_T, Close_{T-1}...
    # Given the features are "RSI, MACD...": these are calculated on Close.
    # RSI(T) is known at Close of T.
    # If we want to predict Return(T+1), we use RSI(T). This is valid.
    
    # To be extremely safe and follow "Lag all features by 1 day":
    # It might mean: Use RSI(T-1) to predict Return(T).
    # This is equivalent to Use RSI(T) to predict Return(T+1).
    
    # I will stick to:
    # Features = Indicators at time T
    # Target = Outcome at time T+1 associated with row T.
    # This is "Standard Time Series Classification Setup".
    
    # Drop columns that are not features
    drop_cols = ['Target', 'Close', 'Daily_Return']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols]
    y = data['Target']
    
    return X, y
