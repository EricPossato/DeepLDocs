import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Parameters you can experiment with
n_samples = 500
class_sep = 1.5
flip_y = 0
random_state_class0 = 42
random_state_class1 = 24

# Generate class 0 with 1 cluster
X0, y0 = make_classification(
    n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, weights=[1.0, 0.0],
    class_sep=class_sep, flip_y=flip_y, random_state=random_state_class0
)

# Generate class 1 with 2 clusters
X1, y1 = make_classification(
    n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=2, n_classes=2, weights=[0.0, 1.0],
    class_sep=class_sep, flip_y=flip_y, random_state=random_state_class1
)

# Combine
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0 (1 cluster)', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1 (2 clusters)', alpha=0.6)
plt.title("Synthetic Dataset: 1 cluster vs 2 clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
#save image
plt.savefig("docs/exercise3/dataset.png")
#plt.show()


