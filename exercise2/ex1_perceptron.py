import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
def generate_data():
    np.random.seed(42)

    # Class 0 (label = -1)
    mean_class0 = [1.5, 1.5]
    cov_class0 = [[0.5, 0], [0, 0.5]]
    class0_samples = np.random.multivariate_normal(mean_class0, cov_class0, 1000)
    labels0 = -1 * np.ones(1000)

    # Class 1 (label = +1)
    mean_class1 = [5, 5]
    cov_class1 = [[0.5, 0], [0, 0.5]]
    class1_samples = np.random.multivariate_normal(mean_class1, cov_class1, 1000)
    labels1 = np.ones(1000)

    # Combine
    X = np.vstack((class0_samples, class1_samples))
    y = np.hstack((labels0, labels1))

    return X, y, class0_samples, class1_samples

# Perceptron implementation
def perceptron_train(X, y, lr=0.01, max_epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)   # weight vector
    b = 0.0                    # bias term
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
w, b, accuracies, final_predictions = perceptron_train(X, y)

# Results
print("\nFinal Results:")
print("Weights:", w)
print("Bias:", b)
print("Final Accuracy:", accuracies[-1])

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = -(w[0] * xx + b) / w[1]  # line equation

plt.figure(figsize=(8, 6))
plt.scatter(class0_samples[:, 0], class0_samples[:, 1], c='blue', alpha=0.5, label="Class 0 (-1)")
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], c='red', alpha=0.5, label="Class 1 (+1)")
plt.plot(xx, yy, 'k--', label="Decision Boundary")

# Misclassified points
misclassified = X[final_predictions != y]
if len(misclassified) > 0:
    plt.scatter(misclassified[:, 0], misclassified[:, 1],
                edgecolors='yellow', facecolors='none', s=80, label="Misclassified")

plt.title("Perceptron Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)

#save the image as a png file
plt.savefig("exercise2_ex1_perceptron.png")

# Plot training accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)
plt.savefig("exercise2_ex1_perceptron_accuracy.png")
