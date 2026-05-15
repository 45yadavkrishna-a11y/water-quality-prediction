import os
import joblib
from src.preprocess import load_and_preprocess
from src.train import train_models

os.makedirs('models', exist_ok=True)

# 1. Load & Preprocess
X_train, X_test, y_train, y_test, scaler = \
    load_and_preprocess('data/water_potability.csv')

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# 2. Train all models
svm, rf, xgb, nn, results = train_models(X_train, X_test, y_train, y_test)

print("\n✅ All models trained and saved successfully!")