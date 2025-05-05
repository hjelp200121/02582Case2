import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def sil_score_test(X):
    sil_scores = []
    for k in range (2,10):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    return sil_scores

def select_cols_to_drop(df, corr_cutoff):
    # continuous columns
    continuous_cols = df.columns[0:51]

    # absolute value of correlation matrix
    corr_matrix = df[continuous_cols].corr().abs()

    # upper triangular array of True booleans
    bool_array = np.ones(corr_matrix.shape).astype(bool)
    T = np.triu(bool_array, 1)

    T_corr_matrix = corr_matrix.where(T)

    # array of columns to drop
    drop_cols = []

    # drop columns that have at least corr_cutoff absolute correlation with at least one other column
    col_arr = T_corr_matrix.columns
    for col in col_arr:
        if any(T_corr_matrix[col] > corr_cutoff):
            T_corr_matrix = T_corr_matrix.drop(columns=col)
            drop_cols.append(col)
    
    return drop_cols

df = pd.read_csv("data/HR_data_cleaned.csv")

drop_cols = select_cols_to_drop(df, corr_cutoff=0.9)

df = df.drop(columns=drop_cols)

feature_cols = list([df.columns[36]] + list(df.columns[38:48]))
X = df[feature_cols]

print(feature_cols)
#sil_score_test(X)

# --- K-Means clustering ---
k = 3
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(X)

df_features = df.copy()
df_features['Cluster'] = labels
cluster_profiles = df_features.groupby('Cluster').mean()
physio_profiles = cluster_profiles[df.columns[:32]]

plt.figure(figsize=(25,12))
for cl in physio_profiles.index:
    plt.plot(
        physio_profiles.loc[cl],
        marker='o',
        label=f"Cluster {cl}"
    )

plt.title("Mean Physiological Measurements per Cluster", fontsize=20)
plt.xlabel("Physiological Feature", fontsize=20)
plt.ylabel("Mean Value", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.legend(title="Cluster", fontsize=20)
plt.tight_layout()
plt.grid()
plt.savefig("results/TESTEmotionalToPhysio.pdf")
plt.show()