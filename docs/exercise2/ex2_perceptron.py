import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
def generate_data():
    np.random.seed(42)

    # Class 0 (label = -1)
    mean_class0 = [3, 3]
    cov_class0 = [[1.5, 0], [0, 1.5]]
    class0_samples = np.random.multivariate_normal(mean_class0, cov_class0, 1000)
    labels0 = -1 * np.ones(1000)

    # Class 1 (label = +1)
    mean_class1 = [4, 4]
    cov_class1 = [[1.5, 0], [0, 1.5]]
    class1_samples = np.random.multivariate_normal(mean_class1, cov_class1, 1000)
    labels1 = np.ones(1000)

    # Plot the two classes
    plt.figure(figsize=(8, 6))
    plt.scatter(class0_samples[:, 0], class0_samples[:, 1],
                c='blue', alpha=0.5, label="Class 0")
    plt.scatter(class1_samples[:, 0], class1_samples[:, 1],
                c='red', alpha=0.5, label="Class 1")
    plt.title("2D Data Points from Two Classes (Multivariate Normal)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    #plt.show()
    # Save the image as a png file
    plt.savefig("exercise2_ex2_plot.png")

    # Combine
    X = np.vstack((class0_samples, class1_samples))
    y = np.hstack((labels0, labels1))

    return X, y, class0_samples, class1_samples

    

# Perceptron implementation
def perceptron_train(X, y, lr=0.01, max_epochs=100, random_init=False):
    n_samples, n_features = X.shape
    # Initialize weights randomly or zeros
    if random_init:
        w = np.random.randn(n_features) * 0.01
        b = np.random.randn() * 0.01
    else:
        w = np.zeros(n_features)
        b = 0.0           # bias term
    accuracies = []

    for epoch in range(max_epochs):
        errors = 0
        for i in range(n_samples):
            activation = np.dot(w, X[i]) + b
            y_pred = 1 if activation >= 0 else -1
            if y_pred != y[i]:
                # Update rule
                w += lr * y[i] * X[i]
                b += lr * y[i]
                errors += 1

        # Track accuracy
        activation = np.dot(X, w) + b
        predictions = np.where(activation >= 0, 1, -1)
        acc = np.mean(predictions == y)
        accuracies.append(acc)

        if errors == 0:  # converged
            print(f"Converged after {epoch+1} epochs.")
            break

    return w, b, accuracies, predictions

# Generate data
X, y, class0_samples, class1_samples = generate_data()

# Train perceptron
all_runs_acc = []
best_run = None

for run in range(5):
    w, b, accuracies, final_predictions = perceptron_train(X, y, lr=0.01, max_epochs=100, random_init=True)
    final_acc = accuracies[-1]
    all_runs_acc.append(final_acc)

    if best_run is None or final_acc > best_run["acc"]:
        best_run = {"w": w, "b": b, "acc": final_acc, "acc_list": accuracies, "pred": final_predictions}


# Results
print("\nSummary over 5 runs:")
print("Accuracies per run:", all_runs_acc)
print("Average accuracy:", np.mean(all_runs_acc))
print("Best accuracy:", best_run["acc"])
print("Best weights:", best_run["w"])
print("Best bias:", best_run["b"])



# Plot decision boundary of best run
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = -(best_run["w"][0] * xx + best_run["b"]) / best_run["w"][1]

plt.figure(figsize=(8, 6))
plt.scatter(class0_samples[:, 0], class0_samples[:, 1], 
            c='blue', alpha=0.5, label="Class 0 (-1)")
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], 
            c='red', alpha=0.5, label="Class 1 (+1)")
plt.plot(xx, yy, 'k--', label="Decision Boundary")

# Misclassified points separated by class
misclassified = X[best_run["pred"] != y]
misclassified_labels = y[best_run["pred"] != y]

# Class 0 misclassified (true label = -1)
plt.scatter(misclassified[misclassified_labels == -1][:, 0],
            misclassified[misclassified_labels == -1][:, 1],
            c='green', alpha=0.5, label="Misclassified Class 0")

# Class 1 misclassified (true label = +1)
plt.scatter(misclassified[misclassified_labels == 1][:, 0],
            misclassified[misclassified_labels == 1][:, 1],
            c='yellow', alpha=0.5, label="Misclassified Class 1")

plt.title("Best Perceptron Decision Boundary (Overlapping Data)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.savefig("exercise2_ex2_perceptron.png")

# Accuracy progression of best run
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(best_run["acc_list"]) + 1), best_run["acc_list"], marker='o')
plt.title("Training Accuracy over Epochs (Best Run)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)
#save
plt.savefig("exercise2_ex2_perceptron_accuracy.png")