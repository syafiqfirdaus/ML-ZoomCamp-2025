import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def preprocess_data(input_file='asean_market_data.csv', output_file='processed_data.csv'):
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    features = []
    
    for ticker in df.columns:
        price = df[ticker]
        
        # Returns
        ret_1d = price.pct_change(1)
        ret_5d = price.pct_change(5)
        ret_20d = price.pct_change(20)
        
        # Volatility
        vol_20d = ret_1d.rolling(window=20).std()
        
        # Technicals
        rsi = compute_rsi(price)
        macd, macd_signal = compute_macd(price)
        
        # Target: Next 20 days return > 0 (Binary Classification)
        # Shift -20 to align "future return" with "current features"
        future_ret_20d = price.pct_change(20).shift(-20)
        target = (future_ret_20d > 0).astype(int)
        
        # Create DataFrame for this ticker
        ticker_df = pd.DataFrame({
            'Ticker': ticker,
            'Price': price,
            'Ret_1d': ret_1d,
            'Ret_5d': ret_5d,
            'Ret_20d': ret_20d,
            'Vol_20d': vol_20d,
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'Target': target
        })
        
        features.append(ticker_df)
    
    # Concatenate all
    full_df = pd.concat(features)
    
    # Drop NaN
    full_df = full_df.dropna()
    
    print(f"Processed data shape: {full_df.shape}")
    print(full_df.head())
    
    full_df.to_csv(output_file)
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    preprocess_data()
