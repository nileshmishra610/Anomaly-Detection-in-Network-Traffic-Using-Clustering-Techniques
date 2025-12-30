import os
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

app = Flask(__name__)
MODELS_DIR = r"C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\models"

# Load clustering models and anomaly probabilities
models = {}
anomaly_probs = {}
model_names = ['kmeans', 'dbscan', 'gmm', 'hdbscan'] if HDBSCAN_AVAILABLE else ['kmeans', 'dbscan', 'gmm']
for model_name in model_names:
    try:
        models[model_name] = joblib.load(os.path.join(MODELS_DIR, f'{model_name}.joblib'))
        if model_name == 'kmeans':
            print(f"KMeans centroids dtype: {models[model_name].cluster_centers_.dtype}")
        elif model_name == 'gmm':
            print(f"GMM means dtype: {models[model_name].means_.dtype}")
        elif model_name == 'dbscan':
            print(f"DBSCAN components dtype: {models[model_name].components_.dtype}")
    except FileNotFoundError:
        print(f"Warning: {model_name}.joblib not found. {model_name.upper()} predictions unavailable.")
    try:
        anomaly_probs[model_name] = joblib.load(os.path.join(MODELS_DIR, f'{model_name}_anomaly_prob.joblib'))
    except FileNotFoundError:
        print(f"Warning: {model_name}_anomaly_prob.joblib not found. Using distance-based scores.")
        anomaly_probs[model_name] = None

def dbscan_predict(model, X_new):
    if len(model.components_) == 0:
        return np.array([-1], dtype=np.int32)
    X_new_np = X_new.to_numpy().astype(np.float32) if isinstance(X_new, pd.DataFrame) else X_new.astype(np.float32)
    dists = euclidean_distances(X_new_np, model.components_)
    min_dists = np.min(dists, axis=1)
    closest = np.argmin(dists, axis=1)
    labels = np.where(
        min_dists <= model.eps,
        model.labels_[model.core_sample_indices_[closest]],
        -1
    )
    return labels

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data['model'].lower()
        features = data['features']
        
        if model_name not in models:
            return jsonify({'error': f"Model {model_name} not loaded."}), 400
        
        # define PCA column names
        pca_cols = [f'PC{i+1}' for i in range(10)]
        pca_cols_lower = [f'pc{i+1}' for i in range(10)]
        
        # validate input features
        if not all(col in features for col in pca_cols_lower):
            return jsonify({'error': f"Expected PCA features: {pca_cols_lower}"}), 400
        
        # prepare input data as float32 dataframe
        pca_values = [float(features[col]) for col in pca_cols_lower]
        X_pca = np.array(pca_values, dtype=np.float32).reshape(1, 10)
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, dtype=np.float32)
        
        print(f"Input X_pca: {X_pca}")
        print(f"Input X_pca dtype: {X_pca.dtype}")
        print(f"X_pca_df dtypes: {X_pca_df.dtypes}")
        
        model = models[model_name]
        probs = anomaly_probs[model_name]
        
        # cluster prediction
        if model_name == 'hdbscan' and HDBSCAN_AVAILABLE:
            cluster, _ = hdbscan.approximate_predict(model, X_pca)
            cluster = cluster[0]
        elif model_name == 'dbscan':
            cluster = dbscan_predict(model, X_pca_df)[0]
        else:
            cluster = model.predict(X_pca_df)[0]
        
        # calculate score
        if probs and cluster in probs:
            score = float(probs[cluster])
        else:
            if model_name == "dbscan":
                score = 1.0 if cluster == -1 else 0.0
            else:
                centroids = model.cluster_centers_ if model_name == "kmeans" else model.means_
                centroids = centroids.astype(np.float32)
                print(f"Centroids dtype: {centroids.dtype}")
                score = euclidean_distances(X_pca, [centroids[cluster]])[0, 0]
                score = min(max(score / 100.0, 0.0), 1.0)
        
        return jsonify({
            'prediction': 'anomaly' if score > 0.95 else 'normal',
            'cluster': int(cluster),
            'anomaly_score': float(score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)