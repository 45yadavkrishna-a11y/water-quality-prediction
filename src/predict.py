import pandas as pd
import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hides TF background spam

def load_artifacts():
    """Smart loader: detects if the winning model is Keras or Scikit-Learn."""
    try:
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("Scaler not found. Please run src/train.py first.")

    # Check for Keras Neural Network first
    if os.path.exists('models/best_model.keras'):
        import tensorflow as tf
        model = tf.keras.models.load_model('models/best_model.keras')
        return model, scaler, 'keras'
        
    # Otherwise fall back to standard Scikit-Learn model
    elif os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        return model, scaler, 'sklearn'
    else:
        raise FileNotFoundError("No trained model found in models/ directory.")

def engineer_features(input_dict):
    """Applies the exact same feature engineering used in preprocessing."""
    df = pd.DataFrame([input_dict])
    
    df['Chloramines_to_OC'] = df['Chloramines'] / (df['Organic_carbon'] + 1e-5)
    df['Hardness_to_ph'] = df['Hardness'] / (df['ph'] + 1e-5)
    df['Sulfate_to_Solids'] = df['Sulfate'] / (df['Solids'] + 1e-5)
    
    expected_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
        'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Chloramines_to_OC',
        'Hardness_to_ph', 'Sulfate_to_Solids'
    ]
    return df[expected_cols]

def predict_potability(input_dict):
    """Full prediction pipeline supporting both ML and DL models."""
    model, scaler, model_type = load_artifacts()
    
    df_engineered = engineer_features(input_dict)
    scaled_features = scaler.transform(df_engineered)
    
    # Deep Learning (Keras) Prediction
    if model_type == 'keras':
        probability = model.predict(scaled_features, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
    # Standard ML Prediction
    else:
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else None
        
    return prediction, probability

if __name__ == "__main__":
    # Test with pure water data
    sample_data = {
        'ph': 7.1, 'Hardness': 150.0, 'Solids': 12000.0, 'Chloramines': 8.0,
        'Sulfate': 300.0, 'Conductivity': 350.0, 'Organic_carbon': 12.0,
        'Trihalomethanes': 50.0, 'Turbidity': 3.5
    }
    
    pred, prob = predict_potability(sample_data)
    print(f"Prediction: {'Safe (1)' if pred == 1 else 'Unsafe (0)'}")
    if prob is not None:
        print(f"Confidence (Probability of being Safe): {prob:.2%}")