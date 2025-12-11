import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(file_path='processed_data.csv'):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    # Assume 'Date' column exists from index reset or check column 0
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    return df

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model():
    input_file = 'processed_data.csv'
    if not os.path.exists(input_file):
        print(f"{input_file} not found. Run preprocess.py first.")
        return

    df = load_data(input_file)
    
    # Select Features
    # Exclude 'Target', 'Date', 'Ticker', 'Price'
    feature_cols = ['Ret_1d', 'Ret_5d', 'Ret_20d', 'Vol_20d', 'RSI', 'MACD', 'MACD_Signal']
    target_col = 'Target'
    
    # Filter by date for Train/Test split (Time Series Split)
    # Let's say we train on data before 2023, test on 2023
    # Or just simple 80/20 split based on index if data is sorted
    # Since we have multiple tickers, this is tricky. We should split by time.
    
    dates = np.sort(df['Date'].unique())
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, 'scaler.bin')
    
    # Build Model (Simple Nueral Network given tabular input without sequence for now, 
    # or we can use LSTM if we reshape. Let's start with Dense NN as features already have "history" like Vol_20d)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint]
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model in .h5 as well for legacy support if needed
    model.save('model.h5')

if __name__ == "__main__":
    train_model()
