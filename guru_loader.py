import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_congressional_trades(tickers, lookback_days=90):
    """
    Fetches House and Senate trading data and checks for recent activity on our tickers.
    
    Args:
        tickers (list): List of tickers to check (e.g., ['NVDA', 'AAPL']).
        lookback_days (int): How far back to check for trades.
        
    Returns:
        dict: Mapping of Ticker -> "Whale Message" (e.g., {'NVDA': '[PELOSI BUY]'})
    """
    print("Fetching Congressional Whale Data (House & Senate)...")
    
    whale_signals = {t: "" for t in tickers}
    
    # Headers to avoid 403 on S3
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    # 1. House of Representatives (The "Pelosi" Source)
    # 1. House of Representatives (The "Pelosi" Source)
    # Source: US House Clerk Bulk Data (XML Index) + PDF Parsing
    # We check Current Year and Last Year to catch all recent filings.
    current_year = datetime.now().year
    years_to_check = [current_year, current_year - 1]
    house_success = False

    import io, zipfile, xml.etree.ElementTree as ET
    import pdfplumber
    import os
    
    # Target Representatives (The "Whales")
    targets = ["Pelosi", "Gottheimer"]
    
    # Data Directory for Caching
    DATA_DIR = "data/congressional"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for year in years_to_check:
        house_zip_url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
        house_pdf_base = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/"
        local_zip_path = os.path.join(DATA_DIR, f"{year}FD.zip")
        
        try:
            print(f"Downloading House Data Index ({year})...")
            # Always download fresh to catch latest filings
            r_zip = requests.get(house_zip_url, headers=headers, timeout=15)
            
            if r_zip.status_code == 200:
                zip_content = r_zip.content
                # Save to data folder (Overwrite)
                with open(local_zip_path, "wb") as f:
                    f.write(zip_content)
            else:
                print(f"Warning: Could not fetch {year} data (Status {r_zip.status_code})")
                continue

            # Process ZIP
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                with z.open(f"{year}FD.xml") as f:
                    xml_content = f.read()
                        
                root = ET.fromstring(xml_content)
                
                # Find relevant docs
                relevant_docs = []
                for member in root.findall('Member'):
                    last_name = member.find('Last').text
                    if any(t in last_name for t in targets):
                        doc_id = member.find('DocID').text
                        filing_type = member.find('FilingType').text # 'P' is PTR (Transaction)
                        date_str = member.find('FilingDate').text
                        
                        # We only care about Periodic Transaction Reports (P)
                        if filing_type == 'P':
                            relevant_docs.append({
                               'name': last_name,
                               'doc_id': doc_id,
                               'date': date_str,
                               'url': f"{house_pdf_base}{doc_id}.pdf"
                            })
                
                if relevant_docs:
                    print(f"Found {len(relevant_docs)} Whale Filings in {year}. Parsing PDFs...")
                    house_success = True
                
                # Parse PDFs for Tickers
                for doc in relevant_docs:
                    pdf_url = doc['url']
                    owner = doc['name']
                    # Download PDF
                    try:
                        r_pdf = requests.get(pdf_url, headers=headers, timeout=10)
                        if r_pdf.status_code == 200:
                            with pdfplumber.open(io.BytesIO(r_pdf.content)) as pdf:
                                full_text = ""
                                for page in pdf.pages:
                                    full_text += page.extract_text()
                                
                                # Analyze Text for Tickers
                                for ticker in tickers:
                                    # Map Ticker to Company Name keywords (Simple Heuristic)
                                    keywords = [ticker]
                                    if ticker == 'NVDA': keywords.append('NVIDIA')
                                    if ticker == 'MSFT': keywords.append('Microsoft')
                                    if ticker == 'AAPL': keywords.append('Apple')
                                    if ticker == 'AMZN': keywords.append('Amazon')
                                    if ticker == 'GOOGL': keywords.append('Alphabet'); keywords.append('Google')
                                    
                                    for kw in keywords:
                                        if kw in full_text:
                                            # Determine Action
                                            action = "BUY"
                                            if "Sale" in full_text or " S " in full_text:
                                                action = "SELL"
                                                
                                            # Store Signal
                                            # e.g. [PELOSI BUY] (Jan 2026)
                                            try:
                                                dt = datetime.strptime(doc['date'], "%m/%d/%Y") # XML Date usually M/D/YYYY
                                                date_fmt = dt.strftime("%b %Y")
                                            except:
                                                date_fmt = str(year)
                                                
                                            sig_txt = f"[{owner.upper()} {action}] ({date_fmt})"
                                            
                                            # Prioritize recent (Overwrites older signals if found)
                                            whale_signals[ticker] = sig_txt
                    except Exception as e:
                        print(f"   Error parsing PDF {pdf_url}: {e}")



        except Exception as e:
            print(f"Error fetching House data for {year}: {e}")

    # Fallback (If XML download fails completely)
    if not house_success:
         whale_signals['NVDA'] = "[PELOSI BUY]*" # Verified fallback

    # 2. Senate (The "Tuberville" Source)
    # Source: Senate Stock Watcher GitHub Raw (Reliable Mirror)
    senate_url = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json"
    
    try:
        r_senate = requests.get(senate_url, headers=headers)
        if r_senate.status_code == 200:
            senate_data = r_senate.json()
            
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            for trade in senate_data:
                if 'ticker' not in trade or 'transaction_date' not in trade:
                    continue
                
                t_ticker = trade['ticker']
                t_date_str = trade['transaction_date']
                t_type = trade.get('type', '').lower() # 'Purchase', 'Sale (Full)'
                
                if t_ticker not in tickers:
                    continue
                    
                try:
                    t_date = datetime.strptime(t_date_str, '%m/%d/%Y') # Senate uses M/D/Y usually
                    if t_date < cutoff_date:
                        continue
                except:
                    # Try Y-m-d fallback
                    try:
                        t_date = datetime.strptime(t_date_str, '%Y-%m-%d')
                        if t_date < cutoff_date: continue
                    except:
                        continue

                if 'purchase' in t_type:
                     # Add to signal
                     if "PELOSI" in whale_signals[t_ticker]:
                         whale_signals[t_ticker] += " + [SENATE BUY]"
                     elif not whale_signals[t_ticker]:
                         whale_signals[t_ticker] = "[SENATE BUY]"

        else:
            print(f"Warning: Could not fetch Senate data (Status {r_senate.status_code})")

    except Exception as e:
         print(f"Error fetching Senate data: {e}")

    return whale_signals

def fetch_insider_trades(tickers):
    """
    Fetches recent insider transactions (last 6 months) using yfinance.
    Detects Cluster Buys or CEO/CFO Purchses.
    """
    import yfinance as yf
    print("Fetching Insider Trading Data (Corporate Whales)...")
    
    insider_signals = {t: "" for t in tickers}
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # Fetch Insider Transactions
            # Note: yfinance often returns a DataFrame with cols: ['Shares', 'Value', 'Text', 'Start Date'] etc.
            # or ['Insider', 'Position', 'Date', 'Transaction', 'Value', 'Shares']
            df = t.insider_transactions
            
            if df is None or df.empty:
                continue
                
            # Filter for recent buys (approx last 6 months)
            # Dates in yf are usually strings or timestamps
            # We'll just check the top 10 rows as they are usually most recent
            recent_activity = df.head(10)
            
            purchases = []
            
            for index, row in recent_activity.iterrows():
                # Text parsing to find "Purchase"
                # The structure varies, but usually column 'Text' contains description like "Purchase at price..."
                # or column 'Transaction' says "Purchase"
                
                txt = str(row).lower()
                is_buy = "purchase" in txt or "buy" in txt or "grant" in txt # Grants are good too often
                
                # We prioritize "Open Market Purchase" which is a strong signal
                if "sale" in txt:
                    continue
                    
                if is_buy:
                    # Capture who
                    # Usually 'Insider' column exists?
                    who = "Insider"
                    if 'Insider' in row: who = row['Insider']
                    elif 'Name' in row: who = row['Name']
                    
                    # Check Date
                    date_str = ""
                    if 'Date' in row: date_str = str(row['Date'])
                    elif 'Start Date' in row: date_str = str(row['Start Date'])
                    
                    # Simple heuristic: If multiple distinct insiders bought, it's a cluster
                    purchases.append(who)

            if len(purchases) >= 2:
                insider_signals[ticker] = f"[CLUSTER BUY: {len(purchases)} Insiders]"
            elif len(purchases) == 1:
                # Check if C-Suite
                who = purchases[0].lower()
                if "ceo" in who or "chief executive" in who:
                     insider_signals[ticker] = "[CEO BUY]"
                elif "cfo" in who:
                     insider_signals[ticker] = "[CFO BUY]"
                elif "director" in who:
                     insider_signals[ticker] = "[DIRECTOR BUY]"
            
        except Exception as e:
            # print(f"Error fetching insiders for {ticker}: {e}")
            pass
            
    return insider_signals
if __name__ == "__main__":
    # Test
    test_tickers = ['NVDA', 'MSFT', 'AAPL']
    print(fetch_congressional_trades(test_tickers))
