import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

X = df.to_numpy()
headers = df.columns.to_numpy()
positive_mask = [-1,-1,1,-1,1,-1,1,-1,1,1]
happy_val = X[:,38:48]@positive_mask
quantiles = np.quantile(happy_val, [0, 1/3, 2/3, 1])
print(quantiles)

x_happy = X[happy_val > quantiles[2], :]
x_ok = X[(happy_val > quantiles[1]) & (happy_val <= quantiles[2]), :]
x_mad = X[happy_val <= quantiles[1], :]

X_mean = np.mean(X, axis=0)
happy_mean = np.mean(x_happy, axis=0) - X_mean
ok_mean = np.mean(x_ok, axis=0) - X_mean
mad_mean = np.mean(x_mad, axis=0) - X_mean

plotstop = 32

plotcols = list(range(0, 32)) + [36]

plt.figure(figsize=(25,12))
plt.plot(headers[plotcols],happy_mean[plotcols],label=f"{quantiles[2]} < happy value", marker='o')
plt.plot(headers[plotcols],ok_mean[plotcols],label=f"{quantiles[1]} < happy value <= {quantiles[2]}", marker='o')
plt.plot(headers[plotcols],mad_mean[plotcols],label=f"happy value <= {quantiles[1]}", marker='o')
plt.title("Mean per cluster", fontsize=20)
plt.xlabel("Physiological feature", fontsize=20)
plt.ylabel("Mean value", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=20)
plt.legend(title="Cluster", fontsize=20)
plt.tight_layout()
plt.grid()
plt.savefig("results/madratedToPhysio.pdf")
plt.show()