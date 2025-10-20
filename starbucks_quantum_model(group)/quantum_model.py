import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, Model

"""
Hybrid quantum-classical model using PennyLane KerasLayer.

Design choices and contract:
- Input: time-series window of shape (window_size, 1)
- Internals: flatten -> small classical encoder -> quantum layer (AngleEmbedding + StronglyEntanglingLayers)
- Output: 7-step forecast (same contract as original LSTM model)

Notes:
- This implementation uses the default.qubit simulator (no external quantum hardware required).
- For large windows we reduce dimensionality before the quantum layer (classical compression).
"""


def build_quantum_model(input_shape, n_qubits=4, n_q_layers=1):
    """Builds and returns a tf.keras Model with a PennyLane quantum layer.

    Args:
        input_shape: tuple, e.g. (window_size, 1)
        n_qubits: number of qubits in the quantum circuit (small, e.g. 4)
        n_q_layers: number of strongly entangling layers

    Returns:
        A compiled tf.keras.Model that outputs 7 values (7-day forecast).
    """

    # Classical input
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    # Classical encoder to reduce dimensionality to match quantum inputs
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(n_qubits, activation='tanh')(x)  # produce angles in [-1,1]

    # Quantum circuit definition
    dev = qml.device('default.qubit', wires=n_qubits)

    weight_shapes = {'weights': (n_q_layers, n_qubits, 3)}

    @qml.qnode(dev, interface='tf')
    def qcircuit(inputs_q, weights):
        # Angle embedding: map the n_qubits inputs to rotations
        qml.templates.AngleEmbedding(inputs_q, wires=list(range(n_qubits)))
        # Variational entangling layers
        qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
        # Return expectation values for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Wrap the quantum circuit as a Keras layer
    qlayer = qml.qnn.KerasLayer(qcircuit, weight_shapes, output_dim=n_qubits)

    q_out = qlayer(x)

    # Post-quantum classical processing to produce 7 outputs
    y = layers.Dense(32, activation='relu')(q_out)
    outputs = layers.Dense(7)(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


if __name__ == '__main__':
    # Quick sanity build when run directly
    m = build_quantum_model((365, 1))
    m.summary()
