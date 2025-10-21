import os
import joblib
import numpy as np

MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"


def load_model_and_scaler():
    """Rebuilds the hybrid model architecture and loads saved weights plus scaler.

    The training process saves only model weights and a small meta file containing
    the window size. We rebuild the architecture using the same defaults and
    then load the weights.
    """
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Read meta
    meta_path = os.path.join(MODEL_DIR, MODEL_NAME + ".meta")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            window_size = int(f.read().strip())
    else:
        window_size = 365

    # Rebuild model (quantum-only)
    model = build_quantum_model(window_size)

    # Load weights if available
    weights_path = os.path.join(MODEL_DIR, MODEL_NAME + ".weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    return model, scaler

def prepare_input_for_inference(recent_data, window_size, scaler):
    recent_data_array = np.array(recent_data).reshape(-1, 1)
    scaled_data = scaler.transform(recent_data_array)
    
    X_input = scaled_data.reshape(1, window_size, 1)
    return X_input


def predict_n_days(model, scaler, recent_data, window_size, n_days):
    """Iteratively predict n_days into the future.

    The model was trained to output 7 steps at once (Dense(7)). To allow arbitrary
    n_days, we iteratively predict and append the predictions (in scaled space) to
    the rolling input window.
    """
    if n_days <= 0:
        return []

   
    recent_array = np.array(recent_data).reshape(-1, 1)
    scaled = scaler.transform(recent_array).flatten().tolist()

    preds = []
    while len(preds) < n_days:
       
        input_window = np.array(scaled[-window_size:]).reshape(1, window_size, 1)
        scaled_out = model.predict(input_window)
       
        scaled_out_flat = scaled_out.flatten().tolist()

        for val in scaled_out_flat:
            if len(preds) >= n_days:
                break
            preds.append(val)
            
            scaled.append(val)

   
    pred_arr = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(pred_arr).flatten().tolist()
    return inv


def build_quantum_model(window_size: int):
    """Build a pure quantum model that predicts 7 days from a recent window.

    Implementation notes:
    - Minimal classical preprocessing: flatten the window to a vector of size
      equal to the number of qubits (or reduce via a Dense projection).
    - A PennyLane QNode processes the inputs and returns expectations which
      a small Dense head maps to 7 predictions.

    This function raises an ImportError if PennyLane is not installed.
    """
    # Import PennyLane and TensorFlow only when building the quantum model to
    # avoid forcing these heavy dependencies on other utilities.
    try:
        import pennylane as qml
        from pennylane.qnn import KerasLayer
    except Exception as e:
        raise ImportError("PennyLane is required for the quantum model. Install it with `pip install pennylane`.\n" + str(e))

    # Build a TF Keras model where the 'quantum' part is a KerasLayer wrapping a QNode.
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Flatten

    # Choose number of qubits based on the window size; clamp for simulator speed
    n_qubits = min(12, max(4, window_size // 30))

    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    n_layers = 2
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    qnode = qml.QNode(circuit, dev, interface="tf")
    qlayer = KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    input_layer = Input(shape=(window_size, 1))
    x = Flatten()(input_layer)
    x = Dense(n_qubits, activation="tanh")(x)
    x = qlayer(x)
    out = Dense(7)(x)

    model = Model(inputs=input_layer, outputs=out)
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
