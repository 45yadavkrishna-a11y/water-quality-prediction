import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass distribution:\n", df['Potability'].value_counts())

    df['ph'] = df['ph'].fillna(df['ph'].median())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler