import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/HR_data_cleaned.csv")

physio_cols = df.columns[0:52] 
questionnaire_cols = [df.columns[55]] + list(df.columns[57:67])
feature_cols = list(physio_cols) + list(questionnaire_cols)

X = df[feature_cols]
phase = df[df.columns[52]]
participant_ids = df[df.columns[53]]
rounds = df[df.columns[51]]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --- K-Means clustering ---
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.2f}")

# --- Analyze clusters by phase ---
cluster_phase_df = pd.DataFrame({
    'Cluster': labels,
    'Phase': phase.values
})
phase_counts = pd.crosstab(cluster_phase_df['Cluster'], cluster_phase_df['Phase'])
print("\nCluster distribution across phases:")
print(phase_counts)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", s=60)
plt.title("K-Means Clusters (PCA-Reduced Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

df_features = df[feature_cols].copy()
df_features['Cluster'] = labels
cluster_profiles = df_features.groupby('Cluster').mean()
print("=== Cluster Profiles (mean values per feature) ===")
print(cluster_profiles)
questionnaire_cols = [
    'Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 
    'inspired', 'nervous', 'determined', 'attentive', 
    'afraid', 'active'
]

q_profiles = df_features.groupby('Cluster')[questionnaire_cols].mean()
print("\n=== Questionnaire Profiles by Cluster ===")
print(q_profiles)

# 4. Visualize questionnaire profiles
plt.figure(figsize=(8, 5))
for cl in q_profiles.index:
    plt.plot(
        questionnaire_cols,
        q_profiles.loc[cl],
        marker='o',
        label=f"Cluster {cl}"
    )
plt.title("Mean Questionnaire Responses per Cluster")
plt.xlabel("Questionnaire Item")
plt.ylabel("Mean Score")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()


cluster_profiles = df_features.groupby('Cluster').mean()
global_mean = df_features[feature_cols].mean()
diff_from_mean = cluster_profiles - global_mean
top_n = 5
cluster_to_plot = 2  #Choose cluster to take top 5 from

top_features = diff_from_mean.loc[cluster_to_plot].abs().sort_values(ascending=False).head(top_n).index.tolist()
fig, axes = plt.subplots(1, top_n, figsize=(4*top_n, 5), sharey=False)
for i, feature in enumerate(top_features):
    sns.boxplot(data=df_features, x='Cluster', y=feature, ax=axes[i], palette='Set2')
    axes[i].set_title(f"{feature}\n(Cluster {cluster_to_plot} standout)")
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel("")

plt.tight_layout()
plt.suptitle(f"Top {top_n} Distinguishing Features for Cluster {cluster_to_plot}", fontsize=14, y=1.05)
plt.show()