import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# set experimen seed
np.random.seed(42)

# number of samples
n_a = 500
n_b = 500

# Parameters for Class A
mu_A = np.array([0, 0, 0, 0, 0])
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

# Parameters for Class B
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2,  0.4, 1.5, 0.6, 0.0],
    [0.0,  0.0, 0.6, 1.5, 0.3],
    [0.0,  0.0, 0.0, 0.3, 1.5],
])

# Generate Data
A = np.random.multivariate_normal(mean=mu_A, cov=Sigma_A, size=n_a)
B = np.random.multivariate_normal(mean=mu_B, cov=Sigma_B, size=n_b)

# Combine into one dataset
X = np.vstack([A, B])
y = np.array([0] * n_a + [1] * n_b)  # 0 = Class A, 1 = Class B

# Put into DataFrame
df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(5)])
df["class"] = y


# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(df[[f"x{i+1}" for i in range(5)]])

# PCA projection to 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)
evr = pca.explained_variance_ratio_ 

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.6, label="Class A")
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.6, label="Class B")
plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
plt.title("Exercise 2: PCA projection to 2D (Class A vs Class B)")
plt.legend()
plt.tight_layout()
plt.savefig("exercise2_pca_scatter.png", dpi=150, bbox_inches="tight")
print("Saved PCA figure to exercise2_pca_scatter.png")
plt.show()