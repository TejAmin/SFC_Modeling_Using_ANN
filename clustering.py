# ✅ Step 2: Clustering (step2_clustering.py)

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("narx_full_preprocessed_dataset.csv")

# Group by trajectory and compute average particle sizes
avg_features = df.groupby("trajectory_id")[["d10_target", "d50_target", "d90_target"]].mean()

# Run KMeans clustering
n_clusters = 3  # You can experiment with different values
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(avg_features)

# Map cluster labels back to original dataset
cluster_map = dict(zip(avg_features.index, cluster_labels))
df["cluster_id"] = df["trajectory_id"].map(cluster_map)

# Save final clustered dataset
df.to_csv("narx_clustered_dataset.csv", index=False)
print("✅ Clustering complete. Saved to 'narx_clustered_dataset.csv'")

# Optional: Plot clusters to visualize grouping
sns.scatterplot(
    x=avg_features["d50_target"],
    y=avg_features["d90_target"],
    hue=cluster_labels,
    palette="Set2",
    s=100
)
plt.xlabel("Average d50")
plt.ylabel("Average d90")
plt.title("Trajectory Clustering Based on Particle Size")
plt.grid(True)
plt.show()

