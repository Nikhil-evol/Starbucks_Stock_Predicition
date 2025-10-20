import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from quantum_model import build_quantum_model

MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"

def download_stock_data(ticker="SBUX", period="7y"):  
    df = yf.download(ticker, period=period, interval="1d")
    df = df[['Close']]

    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_dates)
    df.ffill(inplace=True)
    return df

def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size - 7 + 1):
        
        X.append(series[i:(i + window_size)])
        
        y_seq = series[(i + window_size):(i + window_size + 7)].reshape(-1)
        y.append(y_seq)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(7)  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def main():
    df = download_stock_data(ticker="SBUX", period="5y")
    pd.DataFrame(df).to_csv('SBUX.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
   
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    window_size = 365  
    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Insufficient data for training set.")
    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("Insufficient data for testing set.")
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Use a hybrid quantum-classical model instead of the LSTM
    try:
        model = build_quantum_model((window_size, 1), n_qubits=6, n_q_layers=2)
    except Exception as e:
        print("Failed to build quantum model, falling back to classical LSTM. Error:", e)
        model = build_lstm_model((window_size, 1))
   
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),  callbacks=[tensorboard_callback])

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
