import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pennylane as qml
from pennylane import numpy as pnp

MODEL_DIR = "saved_models"
MODEL_NAME = "quantum_stock_model.pkl"

def download_stock_data(ticker="SBUX", period="7y"):
    df = yf.download(ticker, period=period, interval="1d")[["Close"]]
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(all_dates)
    df.ffill(inplace=True)
    return df

def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size - 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def build_quantum_model(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Encode classical data
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        # Variational layers
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Weight shapes for training
    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes

def main():
    df = download_stock_data("SBUX", "5y")
    pd.DataFrame(df).to_csv("SBUX.csv")

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaled_data = scaler.fit_transform(df.values)

    window_size = 4
    n_qubits = window_size
    n_layers = 2

    X, y = create_dataset(scaled_data.flatten(), window_size)
    X = pnp.array(X)
    y = pnp.array(y)

    circuit, weight_shapes = build_quantum_model(n_qubits, n_layers)
    weights = pnp.random.uniform(low=0, high=np.pi, size=weight_shapes["weights"], requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=0.05)
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0
        for xi, yi in zip(X, y):
            def cost_fn(w):
                preds = circuit(xi, w)
                return (pnp.mean(preds) - yi) ** 2

            weights, loss = opt.step_and_cost(cost_fn, weights)
            total_loss += loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(X):.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"weights": weights, "scaler": scaler}, os.path.join(MODEL_DIR, MODEL_NAME))
    print("Quantum model and scaler saved successfully!")
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Close'], label='Original Close Price')
    plt.title("Starbucks Stock Prices (Training Data)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.show()


    print("📊 Raw data sample (first 5 rows):")
    print(df.head())

    print("\n🔹 Scaled data sample (first 10 values):")
    print(scaled_data[:10])

    print("\n🧠 Example training pairs (X → y):")
    for i in range(5):
     print(f"X[{i}] = {X[i]}  -->  y[{i}] = {y[i]}")
    


    

if __name__ == "__main__":
    main()

