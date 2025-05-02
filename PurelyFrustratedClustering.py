import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt("data/HR_data_cleaned.csv", skip_header=1, delimiter=",")
headers = np.genfromtxt("data/HR_data_cleaned.csv", dtype="str", skip_footer=312, delimiter=",")
X_frust_mask = X[:, 55]

x_chill = X[X_frust_mask <= 2, :]
x_mid = X[(X_frust_mask > 2) & (X_frust_mask <= 5), :]
x_frust= X[X_frust_mask > 5, :]

X_mean = np.mean(X, axis=0)
chill_mean = np.mean(x_chill, axis=0)
mid_mean = np.mean(x_mid, axis=0)
frust_mean = np.mean(x_frust, axis=0)

chill_diff = chill_mean - X_mean
mid_diff = mid_mean - X_mean
frust_diff = frust_mean - X_mean

plt.figure(figsize=(25,8))
plt.plot(headers[:51],chill_diff[:51],label="frustration <= 2", marker='o',)
plt.plot(headers[:51],mid_diff[:51],label="2 < frustration <= 5", marker='o',)
plt.plot(headers[:51],frust_diff[:51],label="5 < frustration", marker='o',)
plt.title("Mean deviation per cluster")
plt.xlabel("Physiological feature")
plt.ylabel("Mean deviation")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("results/FrustratedToPhysio.pdf")
plt.show()