import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.metrics.pairwise import pairwise_distances

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# load pca data
train_df = pd.read_csv(os.path.join(RESULTS_DIR, 'train_pca.csv'))
val_df = pd.read_csv(os.path.join(RESULTS_DIR, 'validation_pca.csv'))
test_df = pd.read_csv(os.path.join(RESULTS_DIR, 'test_pca.csv'))

# float32 for all numeric columns
X_train = train_df.drop(columns=['label']).astype(np.float32)
X_val = val_df.drop(columns=['label']).astype(np.float32)
X_test = test_df.drop(columns=['label']).astype(np.float32)

print(f"X_train dtypes: {X_train.dtypes}")
print(f"X_train sample: {X_train.head(2)}")

def dbscan_predict(model, X_new, batch_size=1000):
    try:
        if len(model.components_) == 0:
            return np.full(X_new.shape[0], -1, dtype=np.int32)
        labels = np.full(X_new.shape[0], -1, dtype=np.int32)
        X_new_np = X_new.to_numpy().astype(np.float32) if isinstance(X_new, pd.DataFrame) else X_new.astype(np.float32)
        for i in range(0, X_new_np.shape[0], batch_size):
            batch = X_new_np[i:i + batch_size]
            dists = pairwise_distances(batch, model.components_)
            min_dists = np.min(dists, axis=1)
            closest = np.argmin(dists, axis=1)
            labels[i:i + batch_size] = np.where(
                min_dists <= model.eps,
                model.labels_[model.core_sample_indices_[closest]],
                -1
            )
        return labels
    except MemoryError:
        print("⚠ MemoryError in DBSCAN predict. Skipping val/test predictions.")
        return np.full(X_new.shape[0], -1, dtype=np.int32)


k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
train_df['kmeans_cluster'] = kmeans.fit_predict(X_train)
val_df['kmeans_cluster'] = kmeans.predict(X_val)
test_df['kmeans_cluster'] = kmeans.predict(X_test)
joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans.joblib'))
print(f" KMeans model saved to {MODELS_DIR}/kmeans.joblib")
print(f"KMeans centroids dtype: {kmeans.cluster_centers_.dtype}")


dbscan = DBSCAN(eps=0.5, min_samples=30, n_jobs=-1)
train_df['dbscan_cluster'] = dbscan.fit_predict(X_train)
val_df['dbscan_cluster'] = dbscan_predict(dbscan, X_val, batch_size=1000)
test_df['dbscan_cluster'] = dbscan_predict(dbscan, X_test, batch_size=1000)
joblib.dump(dbscan, os.path.join(MODELS_DIR, 'dbscan.joblib'))
print(f" DBSCAN model saved to {MODELS_DIR}/dbscan.joblib")
print(f"DBSCAN components dtype: {dbscan.components_.dtype}")


gmm = GaussianMixture(n_components=3, covariance_type='tied', n_init=10, random_state=42)
gmm.fit(X_train)
gmm.means_ = gmm.means_.astype(np.float32)
gmm.covariances_ = gmm.covariances_.astype(np.float32)
train_df['gmm_cluster'] = gmm.predict(X_train)
val_df['gmm_cluster'] = gmm.predict(X_val)
test_df['gmm_cluster'] = gmm.predict(X_test)
joblib.dump(gmm, os.path.join(MODELS_DIR, 'gmm.joblib'))
print(f" GMM model saved to {MODELS_DIR}/gmm.joblib")
print(f"GMM means dtype: {gmm.means_.dtype}")

def get_anomaly_prob(df, cluster_col):
    return df.groupby(cluster_col)['label'].mean().to_dict()

kmeans_prob = get_anomaly_prob(train_df, 'kmeans_cluster')
joblib.dump(kmeans_prob, os.path.join(MODELS_DIR, 'kmeans_anomaly_prob.joblib'))
print(f" KMeans anomaly probabilities saved to {MODELS_DIR}/kmeans_anomaly_prob.joblib")

dbscan_prob = get_anomaly_prob(train_df, 'dbscan_cluster')
joblib.dump(dbscan_prob, os.path.join(MODELS_DIR, 'dbscan_anomaly_prob.joblib'))
print(f" DBSCAN anomaly probabilities saved to {MODELS_DIR}/dbscan_anomaly_prob.joblib")

gmm_prob = get_anomaly_prob(train_df, 'gmm_cluster')
joblib.dump(gmm_prob, os.path.join(MODELS_DIR, 'gmm_anomaly_prob.joblib'))
print(f" GMM anomaly probabilities saved to {MODELS_DIR}/gmm_anomaly_prob.joblib")

train_df.to_csv(os.path.join(RESULTS_DIR, 'train_clustered.csv'), index=False)
val_df.to_csv(os.path.join(RESULTS_DIR, 'validation_clustered.csv'), index=False)
test_df.to_csv(os.path.join(RESULTS_DIR, 'test_clustered.csv'), index=False)

print(" Clustering done, models and anomaly probabilities saved in results/models/")