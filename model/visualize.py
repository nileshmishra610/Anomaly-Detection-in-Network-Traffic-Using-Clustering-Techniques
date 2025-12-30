import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = r"C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\splits"
PLOTS_DIR = r"C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        print(f"Loading {filename}")
        return pd.read_csv(path)
    else:
        print(f"Error: {path} not found!")
        return None

train_df = load_csv('train_clustered.csv')
val_df   = load_csv('validation_clustered.csv')
test_df  = load_csv('test_clustered.csv')

def plot_clusters_2d(df, cluster_col, title, filename):
    if df is None or cluster_col not in df.columns:
        print(f"Skipping {title}: Data or column not found")
        return
    X = df.iloc[:, :2]  
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df[cluster_col], palette='tab10', s=30)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f" {filename} saved in {PLOTS_DIR}")

plot_clusters_2d(train_df, 'kmeans_cluster', 'KMeans Train Clusters', 'train_kmeans.png')
plot_clusters_2d(val_df, 'kmeans_cluster', 'KMeans Validation Clusters', 'val_kmeans.png')
plot_clusters_2d(test_df, 'kmeans_cluster', 'KMeans Test Clusters', 'test_kmeans.png')

plot_clusters_2d(train_df, 'dbscan_cluster', 'DBSCAN Train Clusters', 'train_dbscan.png')
plot_clusters_2d(val_df, 'dbscan_cluster', 'DBSCAN Validation Clusters', 'val_dbscan.png')
plot_clusters_2d(test_df, 'dbscan_cluster', 'DBSCAN Test Clusters', 'test_dbscan.png')

plot_clusters_2d(train_df, 'gmm_cluster', 'GMM Train Clusters', 'train_gmm.png')
plot_clusters_2d(val_df, 'gmm_cluster', 'GMM Validation Clusters', 'val_gmm.png')
plot_clusters_2d(test_df, 'gmm_cluster', 'GMM Test Clusters', 'test_gmm.png')

print(" All cluster visualizations saved for KMeans, DBSCAN and GMM")