import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import os
import joblib
from sklearn.metrics.pairwise import euclidean_distances

RESULTS_DIR = r"C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\splits"
MODELS_DIR = r"C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\models"

files = {
    "train": "train_clustered.csv",
    "validation": "validation_clustered.csv",
    "test": "test_clustered.csv"
}

def load_csv(name):
    path = os.path.join(RESULTS_DIR, files[name])
    if os.path.exists(path):
        print(f"Loading {name} data from {path}")
        return pd.read_csv(path)
    else:
        print(f"Error: {path} not found!")
        return None

train_df = load_csv("train")
val_df = load_csv("validation")
test_df = load_csv("test")

kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans.joblib'))
dbscan = joblib.load(os.path.join(MODELS_DIR, 'dbscan.joblib'))
gmm = joblib.load(os.path.join(MODELS_DIR, 'gmm.joblib'))

def compute_anomaly_scores(X, clusters, model, model_name, batch_size=1000):
    if X is None or clusters is None:
        print(f"Error: Invalid input in {model_name}. X or clusters is None")
        return None
    if model_name == "DBSCAN":
        # dbscan: noise points (-1) are anomalies
        scores = np.where(clusters == -1, 1.0, 0.0)
    else:
        # kmeans/gmm: higher distance = more anomalies
        if model_name == "KMeans":
            centroids = model.cluster_centers_
        else:  
            centroids = model.means_
        if X.shape[1] != centroids.shape[1]:
            print(f"Error: Dimension mismatch in {model_name}. X.shape[1]={X.shape[1]}, centroids.shape[1]={centroids.shape[1]}")
            return None
        # convert X to float32 for memory efficiency
        X = X.astype(np.float32)
        scores = np.zeros(X.shape[0], dtype=np.float32)
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if mask.sum() > 0:
                for i in range(0, mask.sum(), batch_size):
                    batch_mask = np.zeros(X.shape[0], dtype=bool)
                    batch_mask[np.where(mask)[0][i:i+batch_size]] = True
                    if batch_mask.sum() > 0:
                        dists = euclidean_distances(X[batch_mask], [centroids[cluster]])
                        scores[batch_mask] = dists.flatten()
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    return scores

def evaluate(X, clusters, model, model_name, name="Clustering"):
    if X is None or clusters is None:
        print(f"{name}: Skipping evaluation, data not available\n")
        return
    # silhouette score
    if len(set(clusters)) > 1:
        score = silhouette_score(X, clusters)
        print(f"{name} Silhouette Score: {score:.4f}")
    else:
        print(f"{name}: Only one cluster, silhouette not defined")
    unique, counts = np.unique(clusters, return_counts=True)
    total = len(clusters)
    print(f"{name} Cluster Counts: {dict(zip(unique, counts))}")
    print(f"{name} Cluster Percentages: {dict(zip(unique, [c/total*100 for c in counts]))}")
    # identify small clusters 
    small_clusters = {u: c for u, c in zip(unique, counts) if c/total < 0.05}  # Less than 5% of data
    print(f"{name} Small Clusters (<5% of data): {small_clusters}")
    # anomaly scores
    scores = compute_anomaly_scores(X, clusters, model, model_name)
    if scores is not None:
        print(f"{name} Anomaly Score Stats: Mean={scores.mean():.4f}, Std={scores.std():.4f}, "
              f"Min={scores.min():.4f}, Max={scores.max():.4f}")
        # identify potential anomalies (top 5% scores)
        threshold = np.percentile(scores, 95)
        n_anomalies = (scores > threshold).sum()
        print(f"{name} Potential Anomalies (top 5% scores): {n_anomalies} ({n_anomalies/total*100:.2f}%)\n")

def get_features(df):
    if df is None:
        return None
    pca_cols = [f'PC{i+1}' for i in range(100)]  # Large range to find all PCA columns
    available_cols = [col for col in pca_cols if col in df.columns]
    if not available_cols:
        print(f"Error: No PCA columns found in {df.columns}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_cols = [col for col in numeric_cols if col not in ['label', 'kmeans_cluster', 'dbscan_cluster', 'gmm_cluster']]
        if not available_cols:
            print(f"Error: No numeric columns available for features: {numeric_cols}")
            return None
        print(f"Fallback: Using numeric columns: {available_cols}")
    else:
        print(f"Found PCA columns: {available_cols}")
    return df[available_cols]

if train_df is not None:
    X = get_features(train_df)
    clusters = train_df.get('kmeans_cluster')
    evaluate(X, clusters, kmeans, "KMeans", "KMeans Train Unsupervised")
if val_df is not None:
    X = get_features(val_df)
    clusters = val_df.get('kmeans_cluster')
    evaluate(X, clusters, kmeans, "KMeans", "KMeans Validation Unsupervised")
if test_df is not None:
    X = get_features(test_df)
    clusters = test_df.get('kmeans_cluster')
    evaluate(X, clusters, kmeans, "KMeans", "KMeans Test Unsupervised")

if train_df is not None:
    X = get_features(train_df)
    clusters = train_df.get('dbscan_cluster')
    evaluate(X, clusters, dbscan, "DBSCAN", "DBSCAN Train Unsupervised")
if val_df is not None:
    X = get_features(val_df)
    clusters = val_df.get('dbscan_cluster')
    evaluate(X, clusters, dbscan, "DBSCAN", "DBSCAN Validation Unsupervised")
if test_df is not None:
    X = get_features(test_df)
    clusters = test_df.get('dbscan_cluster')
    evaluate(X, clusters, dbscan, "DBSCAN", "DBSCAN Test Unsupervised")

if train_df is not None:
    X = get_features(train_df)
    clusters = train_df.get('gmm_cluster')
    evaluate(X, clusters, gmm, "GMM", "GMM Train Unsupervised")
if val_df is not None:
    X = get_features(val_df)
    clusters = val_df.get('gmm_cluster')
    evaluate(X, clusters, gmm, "GMM", "GMM Validation Unsupervised")
if test_df is not None:
    X = get_features(test_df)
    clusters = test_df.get('gmm_cluster')
    evaluate(X, clusters, gmm, "GMM", "GMM Test Unsupervised")

print(" Evaluation completed for KMeans, DBSCAN, and GMM with unsupervised metrics")