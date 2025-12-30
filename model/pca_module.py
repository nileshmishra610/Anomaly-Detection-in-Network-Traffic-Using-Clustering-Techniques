import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'splits')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')

MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# input files
train_file = os.path.join(RESULTS_DIR, 'train_split.csv')
val_file = os.path.join(RESULTS_DIR, 'validation_split.csv')
test_file = os.path.join(RESULTS_DIR, 'test_preprocessed.csv')

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# separate features and labels
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_val = val_df.drop(columns=['label'])
y_val = val_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# reduce dimensions while keeping 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

import joblib
joblib.dump(pca, os.path.join(MODELS_DIR, 'pca_model.joblib'))
print(f" PCA model saved to {MODELS_DIR}/pca_model.joblib")

print(f" PCA done. Original features: {X_train.shape[1]}, Reduced features: {X_train_pca.shape[1]}")

plt.figure(figsize=(8,5))
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', color='blue')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR,'explained_variance.png'), dpi=150)
plt.close()
print(f" Explained variance plot saved in {PLOTS_DIR}")

train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
train_pca_df['label'] = y_train.reset_index(drop=True)
train_pca_df.to_csv(os.path.join(RESULTS_DIR,'train_pca.csv'), index=False)

val_pca_df = pd.DataFrame(X_val_pca, columns=[f'PC{i+1}' for i in range(X_val_pca.shape[1])])
val_pca_df['label'] = y_val.reset_index(drop=True)
val_pca_df.to_csv(os.path.join(RESULTS_DIR,'validation_pca.csv'), index=False)

test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])
test_pca_df['label'] = y_test.reset_index(drop=True)
test_pca_df.to_csv(os.path.join(RESULTS_DIR,'test_pca.csv'), index=False)

print(" PCA-transformed CSVs and model saved in results/splits/ and results/models/")