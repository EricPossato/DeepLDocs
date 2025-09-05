import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of samples per class
n_samples = 100  

# Class parameters: (mean, std)
class_params = {
    0: {"mean": [2, 3], "std": [0.8, 2.5]},
    1: {"mean": [5, 6], "std": [1.2, 1.9]},
    2: {"mean": [8, 1], "std": [0.9, 0.9]},
    3: {"mean": [15, 4], "std": [0.5, 2.0]}
}

# Store all samples
data = []
labels = []

for label, params in class_params.items():
    mean = params["mean"]
    std = params["std"]
    
    # Generate Gaussian distributed samples
    x = np.random.normal(mean[0], std[0], n_samples)
    y = np.random.normal(mean[1], std[1], n_samples)
    
    # Stack into dataset
    samples = np.column_stack((x, y))
    data.append(samples)
    labels.extend([label] * n_samples)

# Combine into full dataset
data = np.vstack(data)
labels = np.array(labels)

# Put into a DataFrame
df = pd.DataFrame(data, columns=["x1", "x2"])
df["class"] = labels

# Visualization
plt.figure(figsize=(8,6))
for label in class_params.keys():
    subset = df[df["class"] == label]
    plt.scatter(subset["x1"], subset["x2"], label=f"Class {label}", alpha=0.6)

plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Synthetic Gaussian Dataset (4 classes)")
# Save image
plt.savefig("synthetic_gaussian_dataset.png")