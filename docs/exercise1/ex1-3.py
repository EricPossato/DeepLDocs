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

# Quick visualization
plt.figure(figsize=(8,6))
for label in class_params.keys():
    subset = df[df["class"] == label]
    plt.scatter(subset["x1"], subset["x2"], label=f"Class {label}", alpha=0.6)

# --- after your scatter code, before plt.savefig(...) ---

# three hand-drawn separators 
plt.plot([6.0, 0.6], [-2.0, 10.0], linestyle="--", linewidth=2, color="k", label="approx. boundary")  # between 0 & 1
plt.plot([4.0, 12.0], [2.0, 5.0], linestyle="--", linewidth=2, color="k")                            # between 1 & 2
plt.plot([12.0, 12.0], [-1, 9], linestyle="--", linewidth=2, color="k")                             # isolate class 3

# Optional: a subtle label so it appears once in legend
plt.legend()

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Synthetic Gaussian Dataset (4 classes) with Approximate Boundaries")
# save image
plt.savefig("synthetic_gaussian_dataset_with_boundaries.png", dpi=150, bbox_inches="tight")