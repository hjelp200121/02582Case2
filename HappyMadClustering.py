import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt("data/HR_data_cleaned.csv", skip_header=1, delimiter=",")
headers = np.genfromtxt("data/HR_data_cleaned.csv", dtype="str", skip_footer=312, delimiter=",")
positive_mask = [-1,-1,1,-1,1,-1,1,-1,1,1]
happy_val = X[:,57:67]@positive_mask
quantiles = np.quantile(happy_val, [0, 1/3, 2/3, 1])
print(quantiles)

x_happy = X[happy_val > quantiles[2], :]
x_ok = X[(happy_val > quantiles[1]) & (happy_val <= quantiles[2]), :]
x_mad = X[happy_val <= quantiles[1], :]

X_mean = np.mean(X, axis=0)
happy_mean = np.mean(x_happy, axis=0)
ok_mean = np.mean(x_ok, axis=0)
mad_mean = np.mean(x_mad, axis=0)

happy_diff = happy_mean - X_mean
ok_diff = ok_mean - X_mean
mad_diff = mad_mean - X_mean

plt.figure(figsize=(25,8))
plt.plot(headers[:51],happy_diff[:51],label=f"{quantiles[2]} < happy value", marker='o',)
plt.plot(headers[:51],ok_diff[:51],label=f"{quantiles[1]} < happy value <= {quantiles[2]}", marker='o',)
plt.plot(headers[:51],mad_diff[:51],label=f"happy value <= {quantiles[1]}", marker='o',)
plt.title("Mean deviation per cluster")
plt.xlabel("Physiological feature")
plt.ylabel("Mean deviation")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("results/madratedToPhysio.pdf")
plt.show()