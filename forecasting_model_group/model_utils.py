import os
import joblib
import numpy as np
import pennylane as qml

MODEL_DIR = "saved_models"
MODEL_NAME = "quantum_stock_model.pkl"

def load_model_and_scaler():
    data = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))
    return data["weights"], data["scaler"]

def build_quantum_circuit(n_qubits, n_layers):
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

def predict_n_days(model, scaler, recent_data, window_size, n_days):
    weights = model
    n_layers = weights.shape[0]
    n_qubits = weights.shape[1]
    circuit = build_quantum_circuit(n_qubits, n_layers)

    preds = []
    recent = np.array(recent_data[-window_size:])
    for _ in range(n_days):
        scaled_input = scaler.transform(recent.reshape(-1, 1)).flatten()
        q_output = circuit(scaled_input, weights)
        pred_scaled = np.mean(q_output)
        pred_val = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(pred_val)
        recent = np.append(recent[1:], pred_val)
    return preds
