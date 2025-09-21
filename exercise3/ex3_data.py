import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate class 0 with 2 clusters
X0, y0 = make_classification(
    n_samples=500, n_features=4, n_informative=4, n_redundant=0,
    n_clusters_per_class=2, n_classes=3, weights=[1.0, 0.0, 0.0],
    random_state=42
)

# Generate class 1 with 3 clusters
X1, y1 = make_classification(
    n_samples=500, n_features=4, n_informative=4, n_redundant=0,
    n_clusters_per_class=3, n_classes=3, weights=[0.0, 1.0, 0.0],
    random_state=24
)

# Generate class 2 with 4 clusters
X2, y2 = make_classification(
    n_samples=500, n_features=4, n_informative=4, n_redundant=0,
    n_clusters_per_class=4, n_classes=3, weights=[0.0, 0.0, 1.0],
    random_state=84
)

# Combine datasets
X = np.vstack((X0, X1, X2))
y = np.hstack((y0, y1, y2))

# Visualization (using first 2 features for plotting)
plt.figure(figsize=(7, 7))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0 (2 clusters)', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1 (3 clusters)', alpha=0.6)
plt.scatter(X[y==2, 0], X[y==2, 1], c='green', label='Class 2 (4 clusters)', alpha=0.6)
plt.title("Synthetic Multi-Class Dataset (First 2 Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# save image
plt.savefig("docs/exercise3/ex3_dataset.png")
plt.show()