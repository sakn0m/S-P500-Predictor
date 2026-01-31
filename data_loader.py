import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, period_years=10, quiet=False):
    """
    Downloads historical adjusted close prices for the given tickers and Macro Data.
    Updated for Phase 4 to support 'quiet' mode for cleaner UI.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years*365)
    
    if not quiet:
        print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
    
    # 1. Fetch Target Tickers
    # Silence yfinance internal prints
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
    
    if data.empty:
        if not quiet: print("Warning: No data downloaded.")
        return pd.DataFrame()
    
    # Standardize Single Ticker
    if len(tickers) == 1:
        if isinstance(data.columns, pd.MultiIndex):
            # Try to extract the single ticker level
            try:
                df = data.xs(tickers[0], axis=1, level=0)
            except:
                df = data # Fallback
        else:
            df = data
    else:
        df = data # Multi-ticker handling logic implies external loop usually
    
    # 2. Fetch Macro Data
    if not quiet:
        print("Downloading Macro Data (^VIX, ^TNX)...")
    
    macro_tickers = ['^VIX', '^TNX']
    macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
    
    # Flatten Macro
    try:
        # VIX
        if ('^VIX', 'Close') in macro_data.columns:
            vix = macro_data[('^VIX', 'Close')]
        elif '^VIX' in macro_data.columns: # Flat
             vix = macro_data['^VIX']
        else:
             # Fallback if structure is weird (sometimes yf changes)
             vix = macro_data.iloc[:, 0] # Dangerous but placeholder

        # TNX
        if ('^TNX', 'Close') in macro_data.columns:
            tnx = macro_data[('^TNX', 'Close')]
        elif '^TNX' in macro_data.columns:
             tnx = macro_data['^TNX']
        else:
            tnx = macro_data.iloc[:, 1]
            
        # Create temp DF to join
        macro_df = pd.DataFrame(index=macro_data.index)
        macro_df['VIX'] = vix
        macro_df['US10Y'] = tnx
        
    except Exception as e:
        if not quiet: print(f"Warning: Macro merge failed: {e}")
        macro_df = pd.DataFrame() # Empty
    
    # 3. Merge
    if not macro_df.empty:
        df_combined = df.join(macro_df).ffill()
    else:
        df_combined = df.ffill()
        
    df_combined = df_combined.dropna()

    if not quiet:
        print(f"Successfully downloaded data. Shape: {df_combined.shape}")
        
    return df_combined

def get_sentiment(ticker_symbol):
    """
    Fetches the latest news for a ticker and calculates a sentiment score.
    Returns: float (average sentiment from -1 to 1)
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    try:
        t = yf.Ticker(ticker_symbol)
        news = t.news
        if not news:
            return 0.0
        
        scores = []
        for article in news:
            title = article.get('content', {}).get('title', "")
            summary = article.get('content', {}).get('summary', "")
            text = f"{title}. {summary}"
            
            # VADER compound score
            vs = analyzer.polarity_scores(text)
            comp_score = vs['compound']
            
            # --- ASYMMETRIC SENTIMENT LOGIC (The "Risk Filter") ---
            # Bad news is real. "Good news with a but" is noise.
            
            risk_words = [
                'uncertainty', 'shock', 'strain', 'plunge', 'investigation', 
                'downgrade', 'miss', 'cut', 'headwinds', 'pressure', 'concern',
                'weakness', 'caution', 'volatility', 'risk'
            ]
            
            text_lower = text.lower()
            
            # Check for risk
            has_risk = any(w in text_lower for w in risk_words)
            
            if comp_score > 0 and has_risk:
                # Optimized logic: "Good news with a 'but' is effectively Neutral."
                # We neutralize it to 0.0 to strip out the false optimism.
                comp_score = 0.0
            
            # Note: Negative scores remain untouched (Bad news is respected).
            
            scores.append(comp_score)
            
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0

def get_fundamentals(ticker_symbol):
    """
    Fetches key fundamental ratios. Handles None values safely.
    Returns: dict
    """
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info
        
        def safe_get(key):
            val = info.get(key)
            if val is None:
                return 0.0
            return float(val)
        
        return {
            "Forward_PE": safe_get('forwardPE'),
            "PEG": safe_get('pegRatio'),
            "Profit_Margin": safe_get('profitMargins'),
            "Price_to_Book": safe_get('priceToBook')
        }
    except:
        return {"Forward_PE": 0.0, "PEG": 0.0, "Profit_Margin": 0.0, "Price_to_Book": 0.0}

if __name__ == "__main__":
    # Test block
    test_tickers = ['^GSPC', 'AAPL', 'MSFT']
    try:
        df = fetch_data(test_tickers, period_years=5)
        print(df.head())
        print(df.tail())
    except Exception as e:
        print(f"Error: {e}")
