# Network Traffic Anomaly Detection using Clustering

## Project Overview
This project implements unsupervised anomaly detection in network traffic using clustering algorithms (KMeans, DBSCAN, GMM) on the NSL-KDD dataset. The system uses PCA for dimensionality reduction and provides both a REST API and an Android application for real-time predictions.

## Team Members
Nilesh Mishra(P25CS0006), Shashank Dxit(P25CS008), Juhi Choudhary(P25CS0005)

## Table of Contents
1. [Dependencies and Requirements](#dependencies-and-requirements)
2. [Dataset Information](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Training Instructions](#training-instructions)
6. [Inference Instructions](#inference-instructions)
7. [Android Application](#android-application)
8. [Key Results](#key-results)
9. [Troubleshooting](#troubleshooting)

## Dependencies and Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries

pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
flask>=2.3.0
requests>=2.30.0

### Installation

pip install -r requirements.txt

## Dataset

### NSL-KDD Dataset
The project uses the NSL-KDD dataset for network intrusion detection.

**Download Instructions:**

**Option 1: From GitHub (Recommended - Easier)**
1. Visit: https://github.com/Jehuty4949/NSL_KDD
2. Click on "Code" → "Download ZIP" or clone the repository:
    git clone https://github.com/Jehuty4949/NSL_KDD.git
   
3. Extract/Copy the following files into the `dataset/` directory:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
   - `Field Names.csv`

**Option 2: From Official Source**
1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download the NSL-KDD dataset
3. Extract the required files into the `dataset/` directory

**Dataset Structure:**
- Training samples: ~125,973 records
- Test samples: ~22,544 records
- Features: 41 network traffic attributes
- Binary classification: Normal vs Anomaly
- Anomaly ratio: ~53% in both train and test sets

## Project Structure

project_root/
|___AndroidApp/
|   |__ AnomalyDetection             #Android App Source Code
│
├── dataset/                         # Dataset directory 
│   ├── KDDTrain+.txt                # Training data
│   ├── KDDTest+.txt                 # Test data
│   └── Field Names.csv              # Feature names
│
├── model/                           # Source code
│   ├── preprocess.py                # Data preprocessing and encoding
│   ├── split_data.py                # Train/validation split
│   ├── pca_module.py                # PCA dimensionality reduction
│   ├── clustering.py                # Clustering algorithms
│   ├── tuning.py                    # Hyperparameter tuning
│   ├── evaluate.py                  # Model evaluation
│   ├── visualize.py                 # Cluster visualizations
│   ├── api.py                       # Flask REST API server
│   └── test_api.py                  # API testing script
│
├── results/                          # Generated outputs
│   ├── splits/                      # Processed data splits (generated)
│   ├── models/                      # Saved models and encoders
│   ├── plots/                       # Visualizations
│   └── tuning/                      # Hyperparameter tuning results
│
├── android_app/                      # Android application (if included)
│   └── screenshots/                 # App screenshots
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── report.pdf                        # Project report (max 4 pages)

## File Descriptions

### Core Scripts

1. **preprocess.py**
   - Loads NSL-KDD dataset from the dataset directory
   - Encodes categorical features (protocol_type, service, flag) using LabelEncoder
   - Scales all numerical features to [0,1] range using MinMaxScaler
   - Creates binary labels: normal=0, anomaly=1
   - Saves preprocessed train and test data to `results/splits/`
   - Saves encoders and scaler to `results/models/`

2. **split_data.py**
   - Splits preprocessed training data into train/validation sets (80/20 ratio)
   - Uses stratified splitting to maintain class distribution
   - Ensures both splits have similar anomaly ratios
   - Saves train_split.csv and validation_split.csv

3. **pca_module.py**
   - Applies PCA for dimensionality reduction
   - Retains 95% of variance automatically
   - Reduces ~41 features to ~10 principal components
   - Transforms train, validation, and test sets
   - Saves PCA model to `results/models/pca_model.joblib`
   - Generates explained variance plot

4. **clustering.py**
   - Implements three clustering algorithms:
     - **KMeans**: k=3 clusters, 50 initializations
     - **DBSCAN**: eps=0.5, min_samples=30, parallel processing
     - **GMM**: 3 components, tied covariance, 10 initializations
   - Fits models on PCA-transformed training data
   - Predicts clusters for validation and test sets
   - Computes anomaly probability for each cluster based on training labels
   - Saves all models and probability mappings to `results/models/`
   - Uses float32 for memory efficiency

5. **tuning.py**
   - Performs comprehensive hyperparameter tuning for all algorithms
   - **KMeans**: Tests k values from 2 to 9
   - **DBSCAN**: Tests eps={0.2, 0.3, 0.4, 0.5, 0.6}, min_samples={5, 10, 20, 30}
   - **GMM**: Tests components=2-9, covariance_type={full, tied, diag, spherical}
   - Evaluates using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores
   - Generates tuning plots and saves best parameter recommendations
   - Saves all results to `results/tuning/`

6. **evaluate.py**
   - Computes unsupervised evaluation metrics for all models
   - Calculates Silhouette scores for cluster quality
   - Analyzes cluster distributions and sizes
   - Computes anomaly scores based on:
     - Distance from cluster centroids (KMeans, GMM)
     - Noise point identification (DBSCAN)
   - Identifies potential anomalies (top 5% anomaly scores)
   - Displays statistics for train, validation, and test sets

7. **visualize.py**
   - Creates 2D scatter plots of clusters using first 2 principal components
   - Generates separate visualizations for train, validation, and test sets
   - Creates plots for all three algorithms (KMeans, DBSCAN, GMM)
   - Saves 9 plots total to `results/plots/`
   - Color-codes clusters for easy interpretation

8. **api.py**
   - Flask REST API server for real-time predictions
   - Endpoint: POST /predict
   - Accepts 10 PCA features (pc1 to pc10) in JSON format
   - Returns cluster assignment and anomaly score
   - Supports all three clustering models (kmeans, dbscan, gmm)
   - Uses float32 for efficient processing
   - Handles errors gracefully with informative messages
   - Runs on localhost:5000 by default

9. **test_api.py**
   - Tests API with sample requests for all three models
   - Demonstrates proper JSON payload format
   - Shows expected response structure
   - Useful for debugging API connectivity

## Training Instructions

### Prerequisites
- Python 3.8+ installed
- Dataset files downloaded and placed in `dataset/` directory
- All dependencies installed via `pip install -r requirements.txt`

### Complete Training Pipeline

Run the following scripts **in order**:

#### Step 1: Preprocess the Data
```bash 
python model/preprocess.py
```
**What it does:**
- Loads raw NSL-KDD data
- Encodes categorical features
- Scales numerical features
- Creates binary labels
- Saves preprocessed data

**Expected output:**
Loaded KDDTrain+.txt with shape (125973, 42)
Last 3 columns: ['dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type']
Loaded KDDTest+.txt with shape (22544, 42)
Last 3 columns: ['dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type']
Detected target column: attack_type
Encoded columns: ['protocol_type', 'service', 'flag']
Detected target column: attack_type
Encoded columns: ['protocol_type', 'service', 'flag']
Train anomalies ratio: 0.4654
Test anomalies ratio: 0.5692
Preprocessing done. Train & Test saved in 'results/splits/'

**Generated files:**
- `results/splits/train_preprocessed.csv`
- `results/splits/test_preprocessed.csv`
- `results/models/label_encoders.joblib`
- `results/models/scaler.joblib`

#### Step 2: Split Training Data 
```bash
python model/split_data.py
```
**What it does:**
- Splits train data into train/validation (80/20)
- Uses stratification to preserve class distribution

**Expected output:**
Loaded preprocessed train data with shape: (125973, 42)
Train label anomaly ratio: 0.47
Validation label anomaly ratio: 0.47
Train split saved: C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\splits\train_split.csv ((100778, 42))
Validation split saved: C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\splits\validation_split.csv ((25195, 42))
Train/validation split completed with stratification check

**Generated files:**
- `results/splits/train_split.csv`
- `results/splits/validation_split.csv`

#### Step 3: Apply PCA Dimensionality Reduction
```bash
python model/pca_module.py
```
**What it does:**
- Applies PCA to reduce dimensions
- Retains 95% of variance
- Transforms all datasets

**Expected output:**
PCA done. Original features: 41, Reduced features: 10
Explained variance plot saved in C:\Users\niles\OneDrive\Desktop\Anomaly Detection in Network Traffic\results\plots       
PCA-transformed CSVs and model saved in results/splits/ and results/models/

**Generated files:**
- `results/models/pca_model.joblib`
- `results/splits/train_pca.csv`
- `results/splits/validation_pca.csv`
- `results/splits/test_pca.csv`
- `results/plots/explained_variance.png`

#### Step 4: Train Clustering Models
```bash
python model/clustering.py
```
**What it does:**
- Trains KMeans, DBSCAN, and GMM on PCA data
- Predicts clusters for all datasets
- Computes anomaly probabilities per cluster

**Expected output:**

KMeans model saved to results/models/kmeans.joblib
DBSCAN model saved to results/models/dbscan.joblib
GMM model saved to results/models/gmm.joblib
KMeans anomaly probabilities saved
DBSCAN anomaly probabilities saved
GMM anomaly probabilities saved
Clustering done, models saved in results/models/


**Generated files:**
- `results/models/kmeans.joblib`
- `results/models/dbscan.joblib`
- `results/models/gmm.joblib`
- `results/models/kmeans_anomaly_prob.joblib`
- `results/models/dbscan_anomaly_prob.joblib`
- `results/models/gmm_anomaly_prob.joblib`
- `results/splits/train_clustered.csv`
- `results/splits/validation_clustered.csv`
- `results/splits/test_clustered.csv`

### Optional Steps

#### Hyperparameter Tuning (Optional)
```bash
python model/tuning.py
```
**What it does:**
- Tests multiple parameter combinations
- Evaluates with unsupervised metrics
- Generates tuning plots

**Runtime:** ~1-3 hours depending on your machine

**Generated files:**
- `results/tuning/clustering_tuning_results.csv`
- `results/tuning/best_parameters.csv`
- `results/tuning/kmeans_silhouette_plot.png`
- `results/tuning/dbscan_tuning_heatmap.png`
- `results/tuning/gmm_tuning_graph.png`

#### Model Evaluation
```bash
python model/evaluate.py
```
**What it does:**
- Computes Silhouette scores
- Analyzes cluster distributions
- Calculates anomaly scores
- Identifies potential anomalies

#### Generate Visualizations
```bash
python model/visualize.py
```
**What it does:**
- Creates 2D cluster scatter plots
- Generates plots for all models and datasets

**Generated files:**
- 9 PNG files in `results/plots/` (3 models × 3 datasets)

### Quick Start (All Steps)
```bash
# Run complete pipeline
python model/preprocess.py && \
python model/split_data.py && \
python model/pca_module.py && \
python model/clustering.py && \
python model/evaluate.py && \
python model/visualize.py
```

**Total runtime:** ~20-40 minutes (excluding tuning)

## Inference Instructions

### Method 1: Using the REST API (Recommended for Android App)

#### Start the API Server
```bash
python model/api.py
`
Server will start on: `http://127.0.0.1:5000`

**Expected output:**
```
 * Running on http://127.0.0.1:5000
```

#### API Endpoint Details

**Endpoint:** `POST /predict`

**Request Format:**
```json
{
  "model": "kmeans",
  "features": {
    "pc1": 0.12,
    "pc2": -0.45,
    "pc3": 0.78,
    "pc4": -0.23,
    "pc5": 0.56,
    "pc6": -0.89,
    "pc7": 0.34,
    "pc8": -0.67,
    "pc9": 0.9,
    "pc10": -0.12
  }
}
```

**Parameters:**
- `model`: String - Must be one of: "kmeans", "dbscan", or "gmm"
- `features`: Object - Contains 10 PCA features (pc1 through pc10)
  - All values must be floats
  - Features must be PCA-transformed (not raw network features)

**Response Format:**
```json
{
  "prediction": "anomaly",
  "cluster": 2,
  "anomaly_score": 0.87
}
```

**Response Fields:**
- `prediction`: String - "anomaly" or "normal" (based on 0.95 threshold)
- `cluster`: Integer - Assigned cluster ID
- `anomaly_score`: Float - Score between 0 and 1 (higher = more anomalous)

#### Test the API
```bash
# In a new terminal (keep api.py running)
python modelrc/test_api.py
```
**Expected output:**
```
Testing KMEANS:
Sending payload: {...}
Status Code: 200
Response: {'prediction': 'normal', 'cluster': 1, 'anomaly_score': 0.34}

Testing DBSCAN:
...

Testing GMM:
...
```

#### API Usage from Android App

The Android application should:
1. Collect 41 raw network traffic features
2. Preprocess features:
   - Encode categorical features using saved label_encoders.joblib
   - Scale features using saved scaler.joblib
   - Transform to PCA space using saved pca_model.joblib
3. Send POST request to API with 10 PCA features
4. Display response (prediction, cluster, score)

**Example Android code (conceptual):**
```java
// After preprocessing and PCA transformation
JSONObject payload = new JSONObject();
payload.put("model", "kmeans");

JSONObject features = new JSONObject();
for (int i = 1; i <= 10; i++) {
    features.put("pc" + i, pcaFeatures[i-1]);
}
payload.put("features", features);

// Send POST request to http://YOUR_SERVER_IP:5000/predict
```


## Android Application

### Overview
The project includes an Android application that integrates with the Flask API for real-time network anomaly detection.

### Features
- **Input Interface**: Accepts network traffic features
- **Real-time Prediction**: Sends data to API and displays results
- **Visualization**: Shows cluster assignments and anomaly scores
- **Multi-model Support**: Switch between KMeans, DBSCAN, and GMM

### Screenshots
See `screenshots/` directory for:
- Input screen (feature entry)
- Results screen (prediction, cluster, score)
- Visualization screen (cluster plots)

### Setup Instructions for Android App

1. **Update API URL in `InputFragment.java`:**

   // Find this line in InputFragment.java
   `private static final String API_URL = "http://127.0.0.1:5000/predict";`
   
   // Replace with your PC's IP address (e.g., http://192.168.1.100:5000/predict)
   `private static final String API_URL = "http://YOUR_PC_IP:5000/predict";`

2. **Preprocessing on Android:**
   - The app must include the saved models (label_encoders.joblib, scaler.joblib, pca_model.joblib)
   - Convert .joblib files to Android-compatible format or use API for full preprocessing

3. **Network Permissions:**
   - Ensure AndroidManifest.xml includes INTERNET permission

## Saved Models

All trained models are saved in `results/models/`:

| File | Description | Size (approx) |
|------|-------------|---------------|
| `kmeans.joblib` | KMeans clustering model (k=3) | ~10 KB |
| `dbscan.joblib` | DBSCAN clustering model | ~50 MB |
| `gmm.joblib` | Gaussian Mixture Model (n=3) | ~10 KB |
| `pca_model.joblib` | PCA transformer (10 components) | ~50 KB |
| `scaler.joblib` | MinMaxScaler for feature scaling | ~5 KB |
| `label_encoders.joblib` | LabelEncoders for categorical features | ~5 KB |
| `kmeans_anomaly_prob.joblib` | Cluster-wise anomaly probabilities | <1 KB |
| `dbscan_anomaly_prob.joblib` | Cluster-wise anomaly probabilities | <1 KB |
| `gmm_anomaly_prob.joblib` | Cluster-wise anomaly probabilities | <1 KB |

**Note:** DBSCAN model is large because it stores all core samples for prediction.

## Key Results

### Dimensionality Reduction
- **Original features:** 41
- **PCA components:** 10
- **Variance retained:** 95%
- **Dimensionality reduction:** 75.6%

### Clustering Performance

#### Unsupervised Metrics
Algorithm,Parameters,Silhouette,Calinski,DaviesBouldin,Score
DBSCAN,"{'eps': 0.5, 'min_samples': 30}",0.4170391,36617.66444114191,1.4366795521704705,2.6421259882991586
GMM,"{'n_components': 3, 'covariance_type': 'tied'}",0.59738094,110951.6870097593,0.7020535296985213,10.990496107423192
KMeans,{'n_clusters': 3},0.5989842,112036.06554550857,0.6972955296194573,11.10529520681235


**Silhouette Scores:**
- KMeans: [0.5989842]
- DBSCAN: [0.4170391]
- GMM: [0.59738094]

#### Cluster Distributions
(Example - update with your results)

**KMeans (k=3):**
- Cluster 0: 35% of data (45% anomalies)
- Cluster 1: 40% of data (55% anomalies)
- Cluster 2: 25% of data (60% anomalies)

**DBSCAN:**
- Core clusters: 92% of data
- Noise points: 8% of data (flagged as anomalies)

**GMM:**
- Similar distribution to KMeans with probabilistic assignments

### Anomaly Detection Strategy

1. **Cluster-based approach:**
   - Small clusters (<5% of data) are considered anomalous
   - Clusters with high anomaly concentration flagged

2. **Distance-based approach:**
   - Samples far from cluster centroids scored higher
   - Top 5% by distance flagged as potential anomalies

3. **DBSCAN-specific:**
   - Noise points (cluster=-1) directly classified as anomalies
   - No additional scoring needed

### Best Hyperparameters (from tuning)

**KMeans:**
- n_clusters: 3
- Reasoning: Optimal silhouette score, interpretable clusters

**DBSCAN:**
- eps: 0.5
- min_samples: 30
- Reasoning: Balance between noise detection and cluster formation

**GMM:**
- n_components: 3
- covariance_type: tied
- Reasoning: Best fit for data distribution

## Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem:** MemoryError during DBSCAN prediction

**Solution:**
- Script already uses batching (1000 samples at a time)
- Reduce batch size in `clustering.py` if needed
- All arrays use float32 to reduce memory

#### 2. Path Errors
**Problem:** FileNotFoundError when loading data/models

**Solution:**
```python
# Update BASE_DIR in scripts if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
Or use absolute paths in `evaluate.py` and `visualize.py`

#### 3. API Connection Refused
**Problem:** Cannot connect to API from Android app

**Solution:**
- Ensure `api.py` is running: `python model/api.py`
- Check firewall settings allow port 5000
- If using external device, replace `127.0.0.1` with your server's IP address
- For Android emulator: use `10.0.2.2` instead of `127.0.0.1`

#### 4. Import Errors
**Problem:** ModuleNotFoundError for packages

**Solution:**
```bash
pip install -r requirements.txt
```

#### 5. Dataset Not Found
**Problem:** Cannot find KDDTrain+.txt

**Solution:**
- Ensure dataset files are in `dataset/` directory
- Check file names match exactly (case-sensitive)
- Download from: https://www.unb.ca/cic/datasets/nsl.html

#### 6. Dimension Mismatch
**Problem:** Error about feature dimension mismatch

**Solution:**
- Ensure input has exactly 10 PCA features for API
- For raw features, must preprocess through full pipeline
- Check that models were trained on same data

### Performance Optimization

#### Speed up training:
- Reduce data size for testing: Use subset of training data
- Reduce PCA components: Change `n_components=0.90` for 90% variance
- Reduce KMeans initializations: Change `n_init=10` instead of 50

#### Reduce memory usage:
- Use float32 instead of float64 (already implemented)
- Process data in batches (already implemented for DBSCAN)
- Delete intermediate CSV files after use

## System Requirements

### Minimum:
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Storage: 2 GB free space
- OS: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)

### Recommended:
- CPU: Quad-core 3.0 GHz
- RAM: 8 GB
- Storage: 5 GB free space
- OS: Latest stable OS version

## Additional Notes

- **Training time:** ~20-30 minutes for complete pipeline (excluding tuning)
- **Inference time:** <1 second per sample via API
- **Model persistence:** All models saved and can be loaded without retraining
- **Clustering approach:** Fully unsupervised (labels used only for evaluation)
- **API scalability:** Single-threaded Flask server (suitable for demo, not production)

## Future Enhancements

1. **Ensemble methods:** Combine predictions from all three models
2. **Multi-class detection:** Identify specific attack types (DoS, Probe, R2L, U2R)
3. **Online learning:** Update models with new data streams
4. **Production API:** Use Gunicorn/uWSGI for better performance
5. **Deep learning:** Implement autoencoder-based anomaly detection
6. **Mobile optimization:** Edge inference without API dependency

## Contact Information
[Add your contact details]


## Acknowledgments
- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- Scikit-learn library
- Flask framework
