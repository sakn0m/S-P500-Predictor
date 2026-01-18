import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, period_years=10):
    """
    Downloads historical adjusted close prices for the given tickers and Macro Data.
    
    Args:
        tickers (list): List of ticker symbols (strings).
        period_years (int): Number of years of history to download.
        
    Returns:
        pd.DataFrame: DataFrame containing Adjusted Close prices for tickers + Macro Data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years*365)
    
    print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
    
    # 1. Fetch Target Tickers
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbols or internet connection.")
    
    # We need OHLCV for advanced indicators (ATR, ADX).
    # yfinance returns MultiIndex if multiple tickers: (Price, Ticker).
    # If single ticker: (Price, ).
    
    # If using yfinance 0.2+, the structure is slightly different.
    # Let's handle the MultiIndex properly.
    
    # If we have a single ticker, we want a DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    # If multiple tickers... proper handling is complex in one table.
    # PROPOSAL: For this pipeline, we iterate through tickers one by one in main loop.
    # So `fetch_data` can standardize on returning a SINGLE Ticker's OHLCV + Macro Data.
    # If we pass multiple tickers, this function might need to return a Dictionary of DataFrames.
    
    # Let's check how 'main.py' uses it.
    # It passes [TICKER]. So single ticker.
    
    if len(tickers) == 1:
        ticker = tickers[0]
        # data might be multi-level with ticker as column level, or flat.
        # Check columns
        if isinstance(data.columns, pd.MultiIndex):
            # Drop level to get simple OHLCV
             df = data.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            df = data
    else:
        # If multiple tickers, we just keep 'Adj Close' for now? NO, we want full data.
        # But we can't flatten efficiently.
        # Let's assume this function handles ONE primary ticker at a time for the "Model Input".
        # OR `main.py` will loop.
        # Let's enforce single ticker or handle specifically.
        # But wait, original request said "list of top constituents".
        pass 

    # For now, let's Stick to the requested Multi-Ticker capability but optimize for the Single-Ticker-at-a-time pipeline usage 
    # since we build the features per ticker.
    
    # Actually, simpler: just return the raw `data` object if >1 ticker,
    # and if ==1 ticker, return the flat OHLCV.
    
    if len(tickers) == 1:
        if isinstance(data.columns, pd.MultiIndex):
            df = data.xs(tickers[0], axis=1, level=1)
        else:
            df = data
    else:
        # Warn that advanced features might need OHLCV which is harder to access in wide format
        # But for 'Price' based features only, simple DF is ok.
        # For Phase 2, let's assume valid OHLCV for the *primary* analysis.
        df = data

    # 2. Fetch Macro Data
    print("Downloading Macro Data (^VIX, ^TNX)...")
    macro_tickers = ['^VIX', '^TNX']
    macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=False)
    
    # Macro data: we just need Close prices typically
    if 'Adj Close' in macro_data.columns:
        macro_df = macro_data['Adj Close']
    elif 'Close' in macro_data.columns:
        macro_df = macro_data['Close']
    else:
        macro_df = macro_data
        
    macro_df = macro_df.rename(columns={'^VIX': 'VIX', '^TNX': 'US10Y'})
    
    # 3. Merge
    # If df is Single Ticker OHLCV (columns like Open, High...), we join.
    # If df is Multi (Adj Close only), we join.
    
    df_combined = df.join(macro_df).ffill()
    df_combined = df_combined.dropna()

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
            scores.append(vs['compound'])
            
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
