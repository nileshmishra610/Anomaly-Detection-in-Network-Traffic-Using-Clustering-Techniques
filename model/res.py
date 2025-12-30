import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
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

def get_true_labels(df):
    if df is None or 'label' not in df.columns:
        print("Error: No 'label' column found for supervised evaluation")
        return None
    # Assume 'normal' is 0 (normal), anything else is 1 (anomaly)
    true_labels = np.where(df['label'] == 'normal', 0, 1)
    return true_labels

def evaluate_unsupervised(X, clusters, model, model_name, name="Clustering"):
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
    return scores

def evaluate_supervised(true_labels, anomaly_scores, model_name, dataset_name, threshold=None):
    if true_labels is None or anomaly_scores is None:
        print(f"Skipping supervised evaluation for {model_name} on {dataset_name}")
        return
    # For DBSCAN, scores are already 0/1
    if model_name == "DBSCAN":
        pred_labels = anomaly_scores.astype(int)  # 1 if anomaly, 0 if normal
    else:
        # For KMeans/GMM, use threshold to binarize scores
        if threshold is None:
            threshold = np.percentile(anomaly_scores, 95)  # Default to 95th percentile as in unsupervised
        pred_labels = np.where(anomaly_scores > threshold, 1, 0)
    
    # Compute metrics
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # ROC-AUC (using continuous scores)
    if np.unique(true_labels).size > 1 and np.unique(anomaly_scores).size > 1:
        auc = roc_auc_score(true_labels, anomaly_scores)
        fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
        print(f"{model_name} {dataset_name} ROC-AUC: {auc:.4f}")
    else:
        auc = None
        print(f"{model_name} {dataset_name}: Cannot compute ROC-AUC (constant labels/scores)")
    
    print(f"{model_name} {dataset_name} Confusion Matrix:\n{conf_matrix}")
    print(f"{model_name} {dataset_name} Precision: {precision:.4f}")
    print(f"{model_name} {dataset_name} Recall: {recall:.4f}")
    print(f"{model_name} {dataset_name} F1-Score: {f1:.4f}\n")
    
    # Optional: Find optimal threshold using validation (if dataset_name == 'Validation')
    if dataset_name == "Validation" and model_name != "DBSCAN":
        optimal_threshold = find_optimal_threshold(true_labels, anomaly_scores)
        print(f"{model_name} Optimal Threshold from Validation: {optimal_threshold:.4f}")
        return optimal_threshold
    return None

def find_optimal_threshold(true_labels, anomaly_scores):
    # Find threshold that maximizes F1-score
    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
    f1_scores = []
    for th in thresholds:
        pred = np.where(anomaly_scores > th, 1, 0)
        f1_scores.append(f1_score(true_labels, pred))
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

# Evaluate for each model and dataset
datasets = {
    "Train": train_df,
    "Validation": val_df,
    "Test": test_df
}
models = {
    "KMeans": ("kmeans_cluster", kmeans),
    "DBSCAN": ("dbscan_cluster", dbscan),
    "GMM": ("gmm_cluster", gmm)
}

optimal_thresholds = {}  # To store optimal thresholds from validation

for model_name, (cluster_col, model) in models.items():
    for dataset_name, df in datasets.items():
        if df is not None:
            X = get_features(df)
            clusters = df.get(cluster_col)
            true_labels = get_true_labels(df)
            scores = evaluate_unsupervised(X, clusters, model, model_name, f"{model_name} {dataset_name} Unsupervised")
            if dataset_name == "Validation":
                optimal_thresholds[model_name] = evaluate_supervised(true_labels, scores, model_name, dataset_name)
            else:
                # Use optimal threshold if available (for test), else default
                th = optimal_thresholds.get(model_name) if dataset_name == "Test" else None
                evaluate_supervised(true_labels, scores, model_name, dataset_name, threshold=th)

print("Evaluation completed for KMeans, DBSCAN, and GMM with both unsupervised and supervised metrics")