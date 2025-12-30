import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_nslkdd(filename, fieldnames_file):
    filepath = os.path.join(DATA_DIR, filename)
    fieldnames_path = os.path.join(DATA_DIR, fieldnames_file)
    
    fieldnames = pd.read_csv(fieldnames_path, header=None)
    cols = fieldnames.iloc[:, 0].tolist()
    
    if cols[-1] != 'attack_type':
        cols.append('attack_type')
    
    # FIXED: usecols=range(42) → Ignore difficulty level (last column)
    df = pd.read_csv(filepath, names=cols, usecols=range(len(cols)))
    
    print(f" Loaded {filename} with shape {df.shape}")
    print(f" Last 3 columns: {list(df.columns[-3:])}")
    return df

def encode_and_scale(df, encoders=None, scaler=None):
    df = df.copy()
    
    label_col = 'attack_type'
    print(f"Detected target column: {label_col}")
    
    y = df[label_col]
    X = df.drop(columns=[label_col])

    if encoders is None:
        encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    else:
        for col, le in encoders.items():
            X[col] = X[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    
    print(f"Encoded columns: {list(encoders.keys())}")
    
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    y_binary = (y != 'normal').astype(int)
    
    return X_scaled, y_binary, encoders, scaler

if __name__=="__main__":
    train_df = load_nslkdd("KDDTrain+.txt", "Field Names.csv")
    test_df = load_nslkdd("KDDTest+.txt", "Field Names.csv")

    X_train_scaled, y_train, encoders, scaler = encode_and_scale(train_df)

    joblib.dump(encoders, os.path.join(MODELS_DIR,'label_encoders.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR,'scaler.joblib'))

    train_scaled = pd.concat([
        X_train_scaled.reset_index(drop=True),
        y_train.reset_index(drop=True).rename('label')
    ], axis=1)
    train_scaled.to_csv(os.path.join(RESULTS_DIR,'train_preprocessed.csv'), index=False)

    X_test_scaled, y_test, _, _ = encode_and_scale(test_df, encoders=encoders, scaler=scaler)
    test_scaled = pd.concat([
        X_test_scaled.reset_index(drop=True),
        y_test.reset_index(drop=True).rename('label')
    ], axis=1)
    test_scaled.to_csv(os.path.join(RESULTS_DIR,'test_preprocessed.csv'), index=False)

    print(f"Train anomalies ratio: {y_train.mean():.4f}")
    print(f"Test anomalies ratio: {y_test.mean():.4f}")
    print("Preprocessing done. Train & Test saved in 'results/splits/'")