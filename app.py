import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Water Quality Prediction", page_icon="🚰", layout="wide")

# Auto train if models don't exist
if not os.path.exists('models/best_model.pkl'):
    st.info("🔄 First time setup... Training models. Please wait 2-3 minutes...")
    os.makedirs('models', exist_ok=True)
    from src.preprocess import load_and_preprocess
    from src.train import train_models
    X_train, X_test, y_train, y_test, scaler_fit = \
        load_and_preprocess('data/water_potability.csv')
    joblib.dump(scaler_fit, 'models/scaler.pkl')
    svm, rf, xgb, nn, results = train_models(X_train, X_test, y_train, y_test)
    st.success("✅ Models trained successfully! Refreshing...")
    st.rerun()

model  = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def who_safety_check(ph, hardness, solids, chloramines, sulfate,
                     conductivity, organic_carbon, trihalomethanes, turbidity):
    issues = []
    if ph < 6.5 or ph > 8.5:
        issues.append(f"⚠️ pH {ph} is outside safe range (6.5-8.5)")
    if chloramines > 8.0:
        issues.append(f"⚠️ Chloramines {chloramines} exceeds safe limit (8.0 mg/L)")
    if sulfate > 400.0:
        issues.append(f"⚠️ Sulfate {sulfate} exceeds safe limit (400 mg/L)")
    if conductivity > 600.0:
        issues.append(f"⚠️ Conductivity {conductivity} exceeds safe limit (600 uS/cm)")
    if trihalomethanes > 90.0:
        issues.append(f"⚠️ Trihalomethanes {trihalomethanes} exceeds safe limit (90 ug/L)")
    if turbidity > 4.5:
        issues.append(f"⚠️ Turbidity {turbidity} exceeds safe limit (4.5 NTU)")
    if solids > 35000.0:
        issues.append(f"⚠️ TDS {solids} exceeds safe limit (35000 mg/L)")
    if organic_carbon > 18.0:
        issues.append(f"⚠️ Organic Carbon {organic_carbon} exceeds safe limit (18.0 mg/L)")
    return issues

presets = {
    "✅ Safe Water":   [7.2, 190.0, 15000.0, 6.0, 310.0, 380.0, 12.0, 55.0, 3.2],
    "❌ Unsafe Water": [4.5, 290.0, 48000.0, 11.5, 430.0, 660.0, 22.0, 108.0, 6.2],
}

# SIDEBAR
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🔍 Predict Water Quality", "📈 Model Performance"])

# PAGE 1 - PREDICTION
if page == "🔍 Predict Water Quality":
    st.title("🚰 Water Quality Prediction System")
    st.markdown("Predict whether water is **Safe** or **Unsafe** based on chemical properties.")
    st.divider()

    with st.expander("📋 WHO Safe Drinking Water Standards", expanded=True):
        who_data = {
            "Parameter":   ["pH", "Hardness", "Solids (TDS)", "Chloramines", "Sulfate",
                            "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"],
            "Safe Range":  ["6.5-8.5", "<=300", "<=35000", "<=8.0", "<=400",
                            "<=600", "<=18.0", "<=90", "<=4.5"],
            "Health Risk": ["Corrosion/Bitterness", "Scale deposits", "Taste issues",
                            "Eye/nose irritation", "Diarrhea", "Salty taste",
                            "Cancer risk", "Liver problems", "Pathogens"],
        }
        st.table(pd.DataFrame(who_data))

    st.divider()
    st.markdown("### 🔘 Quick Select")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Safe Water Sample", use_container_width=True):
            st.session_state['preset'] = presets["✅ Safe Water"]
            st.session_state['preset_name'] = "✅ Safe Water"
    with col_b:
        if st.button("❌ Unsafe Water Sample", use_container_width=True):
            st.session_state['preset'] = presets["❌ Unsafe Water"]
            st.session_state['preset_name'] = "❌ Unsafe Water"

    vals = st.session_state.get('preset', [7.2, 190.0, 15000.0, 6.0, 310.0, 380.0, 12.0, 55.0, 3.2])
    if 'preset_name' in st.session_state:
        st.info(f"Loaded: **{st.session_state['preset_name']}**")

    st.divider()
    st.markdown("### 🧪 Chemical Properties")
    col1, col2 = st.columns(2)
    with col1:
        ph              = st.number_input("pH Level (Safe: 6.5-8.5)",          min_value=0.0,  max_value=14.0,    value=float(vals[0]), step=0.1)
        hardness        = st.number_input("Hardness (Safe: <=300 mg/L)",        min_value=0.0,  max_value=500.0,   value=float(vals[1]), step=0.1)
        solids          = st.number_input("Solids TDS (Safe: <=35000 mg/L)",    min_value=0.0,  max_value=62000.0, value=float(vals[2]), step=1.0)
        chloramines     = st.number_input("Chloramines (Safe: <=8.0 mg/L)",     min_value=0.0,  max_value=15.0,    value=float(vals[3]), step=0.1)
        sulfate         = st.number_input("Sulfate (Safe: <=400 mg/L)",         min_value=0.0,  max_value=500.0,   value=float(vals[4]), step=0.1)
    with col2:
        conductivity    = st.number_input("Conductivity (Safe: <=600 uS/cm)",   min_value=0.0,  max_value=800.0,   value=float(vals[5]), step=0.1)
        organic_carbon  = st.number_input("Organic Carbon (Safe: <=18.0 mg/L)", min_value=0.0,  max_value=30.0,    value=float(vals[6]), step=0.1)
        trihalomethanes = st.number_input("Trihalomethanes (Safe: <=90 ug/L)",  min_value=0.0,  max_value=130.0,   value=float(vals[7]), step=0.1)
        turbidity       = st.number_input("Turbidity (Safe: <=4.5 NTU)",        min_value=0.0,  max_value=10.0,    value=float(vals[8]), step=0.1)

    st.divider()
    if st.button("🔍 Predict Water Quality", use_container_width=True):
        sample        = np.array([[ph, hardness, solids, chloramines, sulfate,
                                   conductivity, organic_carbon, trihalomethanes, turbidity]])
        sample_scaled = scaler.transform(sample)
        probability   = model.predict_proba(sample_scaled)[0]
        issues        = who_safety_check(ph, hardness, solids, chloramines, sulfate,
                                         conductivity, organic_carbon, trihalomethanes, turbidity)
        prediction    = 0 if issues else 1

        st.divider()
        if prediction == 1:
            st.success("✅ Water is SAFE to drink (Potable)")
        else:
            st.error("❌ Water is UNSAFE to drink (Not Potable)")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Safe Probability",   f"{probability[1]:.2%}")
        with col4:
            st.metric("Unsafe Probability", f"{probability[0]:.2%}")

        st.divider()
        if issues:
            st.markdown("### ⚠️ WHO Violations:")
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ All parameters within WHO safe limits!")

# PAGE 2 - MODEL PERFORMANCE
else:
    st.title("📈 Model Performance")
    st.divider()

    st.markdown("### 🏆 Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(6, 4))
    models     = ['SVM', 'Random Forest', 'XGBoost', 'Neural Network']
    accuracies = [60.52, 64.63, 65.09, 60.67]
    colors     = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']
    bars = ax.bar(models, accuracies, color=colors, width=0.4, edgecolor='black')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc}%', ha='center', fontweight='bold')
    ax.axhline(y=65, color='green', linestyle='--', label='65% line')
    ax.legend()
    st.pyplot(fig)

    st.divider()
    st.markdown("### 📊 Model Metrics")
    metrics = {
        "Model":     ["SVM", "Random Forest", "XGBoost", "Neural Network"],
        "Accuracy":  ["60.52%", "64.63%", "65.09%", "60.67%"],
        "Precision": ["61%", "64%", "64%", "60%"],
        "Recall":    ["61%", "65%", "65%", "61%"],
        "F1-Score":  ["59%", "61%", "62%", "58%"],
        "ROC-AUC":   ["0.587", "0.611", "0.619", "0.580"],
    }
    st.table(pd.DataFrame(metrics))

    st.divider()
    st.markdown("### 📂 Dataset Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", "3,276")
    c2.metric("Safe Samples",  "1,278")
    c3.metric("Unsafe Samples","1,998")
    c4.metric("Features",      "9")