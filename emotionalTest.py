import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("data/HR_data_cleaned.csv")
feature_cols = list([df.columns[55]] + list(df.columns[57:67]))
X = df[feature_cols]

# --- K-Means clustering ---
k = 3
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(X)

df_features = df.copy()
df_features['Cluster'] = labels
cluster_profiles = df_features.groupby('Cluster').mean()
physio_profiles = cluster_profiles[df.columns[:51]]

plt.figure(figsize=(8, 5))
for cl in physio_profiles.index:
    plt.plot(
        physio_profiles.loc[cl],
        marker='o',
        label=f"Cluster {cl}"
    )
plt.title("Mean Physiological Measurements per Cluster")
plt.xlabel("Physiological Feature")
plt.ylabel("Mean Value")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()