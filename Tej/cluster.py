# ----------------------------------------------------------------------
#  clustering helper  – DBSCAN + scaling + PCA visualisation
# ----------------------------------------------------------------------
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def cluster_dbscan(
    df: pd.DataFrame,
    *,
    eps: float = 1.0,
    min_samples: int = 6,
    scale: str = "standard",              # "standard" or "minmax"
    exclude: Optional[List[str]] = None,  # columns NOT to use
    random_state: int = 42,
    pca_plot: bool = True,
) -> Tuple[pd.DataFrame, DBSCAN]:
    """
    Cluster the pre-processed data with DBSCAN.

    Returns
    -------
    (df_out, dbscan)
        df_out  – copy of *df* with a new 'cluster' column
        dbscan  – fitted sklearn DBSCAN instance
    """
    exclude = set(exclude or [])
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in exclude
    ]
    if not feature_cols:
        raise ValueError("No numeric features found for clustering.")

    X = df[feature_cols].values

    # ── scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── DBSCAN ─────────────────────────────────────────────────────────
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1,                         # use all cores
    ).fit(X_scaled)

    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    # silhouette only makes sense if ≥2 clusters (ignore noise)
    mask = labels != -1
    sil_score = (
        silhouette_score(X_scaled[mask], labels[mask]) if n_clusters >= 2 else np.nan
    )

    print(f"DBSCAN  eps={eps}  min_samples={min_samples}")
    print(f"   clusters found  : {n_clusters}")
    print(f"   noise points    : {n_noise}")
    if n_clusters >= 2:
        print(f"   silhouette score: {sil_score:.3f}")
    else:
        print("   silhouette score:  —  (need ≥2 clusters without noise)")

    # ── add labels back to a copy of df ───────────────────────────────
    df_out = df.copy()
    df_out["cluster"] = labels

    # ── (optional) 2-D PCA plot ───────────────────────────────────────
    if pca_plot:
        pca = PCA(n_components=2, random_state=random_state)
        xy = pca.fit_transform(X_scaled)

        cmap = plt.cm.get_cmap("tab10", max(n_clusters, 1))
        colors = [
            "lightgrey" if lab == -1 else cmap(lab) for lab in labels
        ]

        plt.figure(figsize=(8, 6))
        plt.scatter(xy[:, 0], xy[:, 1], c=colors, s=20, edgecolor="k", linewidth=0.3)
        plt.title(f"PCA projection – DBSCAN (eps={eps}, min_samples={min_samples})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        plt.show()

    return df_out, dbscan
