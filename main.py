import pandas as pd
from sklearn.cluster import KMeans

data_path = "data/HR_data_cleaned.csv"
df = pd.read_csv(data_path)

X = df.iloc[:, 1:51]
columns_to_average = list(df.columns[[55]]) + list(df.columns[57:67])
Y = df[columns_to_average]

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(X)

df['cluster'] = clusters
cluster_means = df.groupby('cluster')[columns_to_average].mean()

print(cluster_means)