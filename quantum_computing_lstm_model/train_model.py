import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

# Quantum imports
import pennylane as qml
from pennylane import numpy as pnp

MODEL_DIR = "saved_models"
MODEL_NAME = "hybrid_quantum_lstm_stock_model.keras"

def download_stock_data(ticker="SBUX", period="7y"):
    df = yf.download(ticker, period=period, interval="1d")
    df = df[['Close']]

    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_dates)
    df.ffill(inplace=True)
    return df

def create_dataset(series, window_size, horizon=7):
    """
    series: numpy array shape (N, 1)
    returns X: (samples, window_size, 1) and y: (samples, horizon)
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:(i + window_size)])
        y_seq = series[(i + window_size):(i + window_size + horizon)].reshape(-1)
        y.append(y_seq)
    return np.array(X), np.array(y)

# ---------- Quantum feature extractor ----------

def build_quantum_feature_map(n_qubits, n_layers):
    """
    Builds a PennyLane QNode that takes n_qubits angles as inputs and returns n_qubits expectation values.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # inputs expected length n_qubits (angles in radians)
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes

def reduce_window_to_qubits(window, n_qubits):
    """
    window: shape (window_size, 1) or (window_size,)
    returns a 1D array of length n_qubits by selecting evenly spaced indices
    """
    window = window.flatten()
    idx = np.linspace(0, len(window) - 1, n_qubits).astype(int)
    return window[idx]

def compute_quantum_features_for_dataset(X_windows, circuit, weights, n_qubits):
    """
    X_windows: shape (samples, window_size, 1)
    returns X_q: shape (samples, n_qubits) where each sample is the expectation vector.
    """
    features = []
    for i in range(X_windows.shape[0]):
        window = X_windows[i]  # shape (window_size, 1)
        reduced = reduce_window_to_qubits(window, n_qubits)  # shape (n_qubits,)
        # inputs for circuit should be angles; assume scaled_data in [0,1], map to [0, pi]
        angles = reduced * np.pi
        # PennyLane expects pnp arrays if weights are pnp; ensure types align
        angles_pnp = pnp.array(angles)
        q_out = circuit(angles_pnp, weights)  # returns pnp array of length n_qubits
        features.append(np.array(q_out, dtype=np.float64))
    return np.array(features)  # shape (samples, n_qubits)

# ---------- LSTM model (adjusted to consume quantum features) ----------

def build_lstm_model(input_shape):
    """
    input_shape: (timesteps, features)
    We will use timesteps=1 and features=n_qubits in this design.
    """
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(7)  # predict next 7 days
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# ---------- Main pipeline ----------

def main():
    # 1) Data
    df = download_stock_data(ticker="SBUX", period="5y")
    pd.DataFrame(df).to_csv('SBUX.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)  # shape (N, 1)

    # 2) Train/test split
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # 3) windows
    window_size = 365
    horizon = 7
    X_train_raw, y_train = create_dataset(train_data, window_size, horizon)
    X_test_raw, y_test = create_dataset(test_data, window_size, horizon)

    if X_train_raw.size == 0 or y_train.size == 0:
        raise ValueError("Insufficient data for training set.")
    if X_test_raw.size == 0 or y_test.size == 0:
        raise ValueError("Insufficient data for testing set.")

    # 4) Quantum feature map settings
    n_qubits = 4       # keep small for speed; try 4 or 6
    n_layers = 2

    circuit, weight_shapes = build_quantum_feature_map(n_qubits, n_layers)
    # initialize weights randomly (fixed during LSTM training in this example)
    q_weights = pnp.random.uniform(low=0, high=np.pi, size=weight_shapes['weights'], requires_grad=False)

    # 5) Compute quantum features for each window (this is the "quantum preprocessing")
    print("Computing quantum features for training set (this may take a bit)...")
    X_train_q = compute_quantum_features_for_dataset(X_train_raw, circuit, q_weights, n_qubits)
    print("Computing quantum features for test set...")
    X_test_q = compute_quantum_features_for_dataset(X_test_raw, circuit, q_weights, n_qubits)

    # 6) Prepare shapes for LSTM: we use timesteps=1 and features=n_qubits
    # So reshape (samples, n_qubits) -> (samples, 1, n_qubits)
    X_train = X_train_q.reshape((X_train_q.shape[0], 1, X_train_q.shape[1]))
    X_test = X_test_q.reshape((X_test_q.shape[0], 1, X_test_q.shape[1]))

    # 7) Build & train LSTM
    model = build_lstm_model((1, n_qubits))
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])

    # 8) Save model & scaler (and quantum weights if desired)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # Save quantum weights too (optional)
    joblib.dump(np.array(q_weights), os.path.join(MODEL_DIR, "quantum_weights.npy"))

    print("Hybrid model, scaler, and quantum weights saved successfully.")

if __name__ == "__main__":
    main()
