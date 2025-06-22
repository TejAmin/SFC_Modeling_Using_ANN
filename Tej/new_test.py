import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def clustering_test(df_clean: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
    """
    Perform clustering on the preprocessed (cleaned) DataFrame using KMeans.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Preprocessed data (after tidy_features).
    max_k : int
        Max number of clusters to test for KMeans (to find optimal K).

    Returns
    -------
    df_labeled : pd.DataFrame
        Copy of df_clean with a new 'cluster' column.
    """
    df = df_clean.copy()

    # ── Drop non-numeric (like 'file' column) if any ──
    features = df.select_dtypes(include="number")

    # ── Scale data ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # ── PCA for visualization ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # ── Find optimal K ──
    best_k, best_score = 2, -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"K={k}, Silhouette Score={score:.3f}")
        if score > best_score:
            best_score, best_k = score, k

    # ── Final clustering ──
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['cluster'] = final_kmeans.fit_predict(X_pca)

    # ── Visualize clusters ──
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette="tab10")
    plt.title("KMeans Cluster Visualization (PCA-reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df
def cluster_clean_data_states_only(df_clean: pd.DataFrame, max_k: int = 6) -> pd.DataFrame:
    """
    Cluster using only the state vector: [T_PM, c, d10, d50, d90, T_TM]
    Assumes spread/skew already calculated and d10/d50/d90 removed by tidy_features.

    Returns a DataFrame with 'cluster' labels.
    """
    # Filter only state variables + engineered ones
    state_cols = [col for col in df_clean.columns if col in ['T_PM', 'c', 'T_TM', 'spread', 'skew']]
    df_states = df_clean[state_cols].copy()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_states)

    # PCA for visualization only
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Optimal K selection
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"K={k}, Silhouette Score={score:.3f}")
        if score > best_score:
            best_k = k
            best_score = score

    # Final clustering
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = final_kmeans.fit_predict(X_pca)

    df_result = df_clean.copy()
    df_result['cluster'] = labels

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
    plt.title("Cluster Visualization (State Variables Only)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_result
