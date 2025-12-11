import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data():
    tickers = {
        'Indonesia ETF': 'EIDA.JK',
        'Singapore ETF': 'EWS',
        'Malaysia ETF': 'EWM',
        'Philippines ETF': 'EPHE',
        'Vietnam ETF': 'VNM',
        'CapitaLand Integ. Comm. Trust': 'C38U.SI',
        'Ascendas REIT': 'A17U.SI',
        'DBS Group': 'D05.SI',
        'Maybank': '1155.KL',
        'BCA': 'BBCA.JK'
    }

    start_date = '2015-01-01'
    end_date = '2024-01-01'

    data = {}
    print("Downloading data...")
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[name] = df
            else:
                print(f"Failed to download {ticker} (Empty DF)")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    # Combine Close prices into a single DataFrame
    close_prices = pd.DataFrame()

    for name, df in data.items():
        # Handle MultiIndex columns if present (common in recent yfinance versions)
        try:
            if isinstance(df.columns, pd.MultiIndex):
                # Check where 'Close' is
                if 'Close' in df.columns.get_level_values(0):
                     close = df['Close']
                     if isinstance(close, pd.DataFrame):
                         close = close.iloc[:, 0]
                else:
                     # Fallback
                     close = df.iloc[:, 0] 
            else:
                if 'Close' in df.columns:
                    close = df['Close']
                else:
                    close = df.iloc[:, 0]
        except Exception as e:
             print(f"Error processing {name}: {e}")
             continue
        
        close_prices[name] = close

    # Forward fill missing data
    close_prices = close_prices.ffill().dropna()

    print(f"Data Shape: {close_prices.shape}")
    print(close_prices.head())
    
    output_file = 'asean_market_data.csv'
    close_prices.to_csv(output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    fetch_data()
