Quantum model setup and run

This folder contains a pure-quantum forecasting model using PennyLane and a
TensorFlow Keras wrapper. The implementation expects PennyLane and TensorFlow
to be installed in the Python environment.

Quick setup (PowerShell)

# Create and activate a venv
C:/Users/Asus/AppData/Local/Programs/Python/Python314/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install dependencies
pip install -r .\requirements.txt
pip install tensorflow

# For a smoke test, edit `train_model.py` to set epochs=1, then run:
python .\train_model.py

Notes
- The quantum model uses a PennyLane `KerasLayer` wrapping a QNode. Simulation on
  CPU can be slow for many qubits or layers; the code clamps the number of qubits
  to keep runs reasonable for development.
- The training process saves model weights and a small meta file (window size).
  The loader rebuilds the model using the same defaults and loads weights.
- If PennyLane is not installed, `build_quantum_model` will raise an ImportError
  with instructions to install PennyLane.

If you want me to install dependencies and run a 1-epoch smoke test here, say
"run smoke test" and I'll proceed (I'll install packages into the workspace
Python environment).