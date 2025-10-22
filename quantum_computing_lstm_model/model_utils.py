import os
import joblib
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp

MODEL_DIR = "saved_models"
MODEL_NAME = "hybrid_quantum_lstm_stock_model.keras"  # Make sure this matches your saved model name

# -------- Quantum utilities --------

def build_quantum_feature_map(n_qubits, n_layers):
    """Same feature map as used in training."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def reduce_window_to_qubits(window, n_qubits):
    """Reduce large input window (e.g., 365) to n_qubits points (e.g., 4)."""
    window = window.flatten()
    idx = np.linspace(0, len(window) - 1, n_qubits).astype(int)
    return window[idx]


def quantum_preprocess(window, circuit, weights, scaler, n_qubits):
    """
    Applies quantum feature extraction to one window of raw prices.
    Returns the circuit expectation outputs (classical vector).
    """
    scaled = scaler.transform(window.reshape(-1, 1)).flatten()
    reduced = reduce_window_to_qubits(scaled, n_qubits)
    angles = reduced * np.pi
    q_features = circuit(pnp.array(angles), weights)
    return np.array(q_features, dtype=np.float64)


# -------- Model loading --------

def load_model_and_scaler():
    # First try the local saved_models directory
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    weights_path = os.path.join(MODEL_DIR, "quantum_weights.npy")
    
    # If files don't exist locally, try the forecasting_model_group directory
    if not os.path.exists(model_path):
        alt_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "forecasting_model_group", "saved_models")
        if os.path.exists(alt_model_dir):
            model_path = os.path.join(alt_model_dir, MODEL_NAME)
            scaler_path = os.path.join(alt_model_dir, "scaler.pkl")
            weights_path = os.path.join(alt_model_dir, "quantum_weights.npy")

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise Exception(f"Failed to load scaler: {str(e)}")

    try:
        if os.path.exists(weights_path):
            # Try loading the weights file
            try:
                # First try loading as numpy file
                q_weights = np.load(weights_path, allow_pickle=True)
            except:
                # If that fails, try loading as pickle file
                import pickle
                with open(weights_path, 'rb') as f:
                    q_weights = pickle.load(f)
            
            # Ensure the weights are in the correct format
            if isinstance(q_weights, np.ndarray):
                # Convert to PennyLane array
                q_weights = pnp.array(q_weights)
            else:
                # Initialize with random weights if format is wrong
                n_qubits, n_layers = 4, 2  # Default architecture
                q_weights = pnp.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits))
        else:
            # Initialize with random weights if file doesn't exist
            n_qubits, n_layers = 4, 2  # Default architecture
            q_weights = pnp.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits))
            print("Warning: Quantum weights file not found. Using random initialization.")
    except Exception as e:
        print(f"Warning: Failed to load quantum weights ({str(e)}). Using random initialization.")
        n_qubits, n_layers = 4, 2  # Default architecture
        q_weights = pnp.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits))

    return model, scaler, q_weights


# -------- Inference --------

def predict_n_days(model, scaler, q_weights, recent_data, window_size, n_days,
                   n_qubits=4, n_layers=2):
    """
    Iteratively predicts n_days into the future using quantum preprocessing.
    """
    if n_days <= 0:
        return []

    circuit = build_quantum_feature_map(n_qubits, n_layers)

    # Keep a working list of data
    recent_data_list = list(recent_data)
    preds_scaled = []

    for _ in range(n_days):
        # Take last 'window_size' values as current window
        current_window = np.array(recent_data_list[-window_size:])

        # Quantum feature extraction
        q_features = quantum_preprocess(current_window, circuit, q_weights, scaler, n_qubits)

        # Prepare input for LSTM (timesteps=1, features=n_qubits)
        X_input = q_features.reshape(1, 1, n_qubits)

        # Predict scaled next values (7-step prediction)
        scaled_pred = model.predict(X_input, verbose=0).flatten()

        # Use the first predicted step for iterative forecasting
        next_scaled_value = scaled_pred[0]
        preds_scaled.append(next_scaled_value)

        # Append to rolling window (in *unscaled* form for next iteration)
        next_unscaled = scaler.inverse_transform(np.array([[next_scaled_value]]))[0, 0]
        recent_data_list.append(next_unscaled)

    # Convert final scaled predictions back to prices
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds_scaled).flatten().tolist()
    return preds_unscaled
