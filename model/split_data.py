import pandas as pd
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'splits')

train_file = os.path.join(RESULTS_DIR, 'train_preprocessed.csv')

df = pd.read_csv(train_file)
print(f" Loaded preprocessed train data with shape: {df.shape}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Train label anomaly ratio: {train_df['label'].mean():.2f}")
print(f"Validation label anomaly ratio: {val_df['label'].mean():.2f}")

train_split_file = os.path.join(RESULTS_DIR, 'train_split.csv')
val_split_file = os.path.join(RESULTS_DIR, 'validation_split.csv')

train_df.to_csv(train_split_file, index=False)
val_df.to_csv(val_split_file, index=False)

print(f" Train split saved: {train_split_file} ({train_df.shape})")
print(f" Validation split saved: {val_split_file} ({val_df.shape})")
print(" Train/validation split completed with stratification check")