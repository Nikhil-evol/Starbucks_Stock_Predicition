import os
import joblib
import numpy as np
import tensorflow as tf

MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"

def load_model_and_scaler():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
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

   
    import numpy as _np
    pred_arr = _np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(pred_arr).flatten().tolist()
    return inv
