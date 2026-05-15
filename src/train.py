import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {name} Results")
    print(f"{'='*40}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Unsafe','Safe']))
    return acc

def train_models(X_train, X_test, y_train, y_test):
    results = {}

    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # SVM
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train_bal, y_train_bal)
    results['SVM'] = evaluate_model("SVM", svm, X_test, y_test)
    joblib.dump(svm, 'models/svm_model.pkl')

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)
    results['Random Forest'] = evaluate_model("Random Forest", rf, X_test, y_test)
    joblib.dump(rf, 'models/rf_model.pkl')

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=42)
    xgb.fit(X_train_bal, y_train_bal)
    results['XGBoost'] = evaluate_model("XGBoost", xgb, X_test, y_test)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # Neural Network
    print("\nTraining Neural Network...")
    nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                       activation='relu',
                       solver='adam',
                       max_iter=500,
                       random_state=42)
    nn.fit(X_train_bal, y_train_bal)
    results['Neural Network'] = evaluate_model("Neural Network", nn, X_test, y_test)
    joblib.dump(nn, 'models/nn_model.pkl')

    # Save best model
    best = max(results, key=results.get)
    print(f"\n✅ Best Model: {best} with accuracy: {results[best]:.4f}")
    best_model = {'SVM': svm, 'Random Forest': rf, 'XGBoost': xgb, 'Neural Network': nn}[best]
    joblib.dump(best_model, 'models/best_model.pkl')

    print("\n📊 Model Comparison:")
    for name, acc in results.items():
        print(f"  {name:20s} → {acc:.4f}")

    return svm, rf, xgb, nn, results