import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'splits')
TUNING_DIR = os.path.join(BASE_DIR, 'results', 'tuning')
os.makedirs(TUNING_DIR, exist_ok=True)

# load pca data
train_df = pd.read_csv(os.path.join(RESULTS_DIR, 'train_pca.csv'))
X_train = train_df.drop(columns=['label']).astype(np.float32)

def evaluate_clustering(X, labels):
    """Compute cluster evaluation metrics."""
    if len(set(labels)) <= 1 or (-1 in set(labels) and len(set(labels)) == 2):
        return -1, -1, -1  # Invalid clustering
    sil = silhouette_score(X, labels)
    cal = calinski_harabasz_score(X, labels)
    dav = davies_bouldin_score(X, labels)
    return sil, cal, dav

results = []

print("\n Tuning KMeans...")
k_values = range(2, 10)
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_train)
    sil, cal, dav = evaluate_clustering(X_train, labels)
    results.append(["KMeans", {"n_clusters": k}, sil, cal, dav])


print("\n Tuning DBSCAN...")
eps_values = [0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_values = [5, 10, 20, 30]
for eps, min_samples in product(eps_values, min_samples_values):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_train)
    sil, cal, dav = evaluate_clustering(X_train, labels)
    results.append(["DBSCAN", {"eps": eps, "min_samples": min_samples}, sil, cal, dav])

print("\n Tuning GMM...")
components = range(2, 10)
cov_types = ['full', 'tied', 'diag', 'spherical']
for n_components, cov_type in product(components, cov_types):
    try:
        model = GaussianMixture(n_components=n_components, covariance_type=cov_type, n_init=10, random_state=42)
        labels = model.fit_predict(X_train)
        sil, cal, dav = evaluate_clustering(X_train, labels)
        results.append(["GMM", {"n_components": n_components, "covariance_type": cov_type}, sil, cal, dav])
    except:
        continue

# save sesults
df_results = pd.DataFrame(results, columns=["Algorithm", "Parameters", "Silhouette", "Calinski", "DaviesBouldin"])

# normalize metrics for combined scoring
df_results["Score"] = df_results["Silhouette"] + (df_results["Calinski"] / 10000) - df_results["DaviesBouldin"]

best_results = (
    df_results.groupby("Algorithm")
    .apply(lambda x: x.loc[x["Score"].idxmax()])
    .reset_index(drop=True)
)

df_results.to_csv(os.path.join(TUNING_DIR, "clustering_tuning_results.csv"), index=False)
best_results.to_csv(os.path.join(TUNING_DIR, "best_parameters.csv"), index=False)

print("\n Tuning complete!")
print(f"All results saved in: {TUNING_DIR}")
print("\n Best Parameters:")
print(best_results)

def plot_kmeans(df):
    kmeans_df = df[df["Algorithm"] == "KMeans"].copy()
    kmeans_df["k"] = kmeans_df["Parameters"].apply(lambda x: eval(str(x))["n_clusters"])
    
    plt.figure(figsize=(8, 5))
    plt.plot(kmeans_df["k"], kmeans_df["Silhouette"], marker='o', color='b')
    plt.title("KMeans: Silhouette Score vs Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(os.path.join(TUNING_DIR, "kmeans_silhouette_plot.png"))
    plt.close()


def plot_dbscan(df):
    db_df = df[df["Algorithm"] == "DBSCAN"].copy()
    eps_vals = [eval(str(p))["eps"] for p in db_df["Parameters"]]
    min_samp_vals = [eval(str(p))["min_samples"] for p in db_df["Parameters"]]
    db_df["eps"] = eps_vals
    db_df["min_samples"] = min_samp_vals
    pivot = db_df.pivot_table(values="Silhouette", index="eps", columns="min_samples")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot, cmap="viridis", origin="lower")
    plt.colorbar(label="Silhouette Score")
    plt.title("DBSCAN Silhouette Heatmap")
    plt.xlabel("min_samples")
    plt.ylabel("eps")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.savefig(os.path.join(TUNING_DIR, "dbscan_tuning_heatmap.png"))
    plt.close()

def plot_gmm(df):
    gmm_df = df[df["Algorithm"] == "GMM"].copy()
    gmm_df["n_components"] = gmm_df["Parameters"].apply(lambda x: eval(str(x))["n_components"])
    gmm_df["cov_type"] = gmm_df["Parameters"].apply(lambda x: eval(str(x))["covariance_type"])
    plt.figure(figsize=(8, 5))
    for cov in gmm_df["cov_type"].unique():
        sub = gmm_df[gmm_df["cov_type"] == cov]
        plt.plot(sub["n_components"], sub["Silhouette"], marker='o', label=f"{cov}")
    plt.title("GMM Silhouette vs Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(TUNING_DIR, "gmm_tuning_graph.png"))
    plt.close() 

plot_kmeans(df_results)
plot_dbscan(df_results)
plot_gmm(df_results)

print("\n Graphs saved in results/tuning/:")
print("  kmeans_tuning_graph.png")      
print("  dbscan_tuning_heatmap.png")
print("  gmm_tuning_graph.png")
