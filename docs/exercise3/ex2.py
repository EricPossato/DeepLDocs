import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Binary cross-entropy loss
def binary_cross_entropy(y, y_hat):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# Derivative of BCE wrt output pre-activation
def bce_derivative(y, y_hat):
    return (y_hat - y) / (y_hat * (1 - y_hat) + 1e-9)

# Generate subset for class 0 with 1 cluster
X0, y0 = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, weights=[1.0, 0.0],
    class_sep=1.5, flip_y=0, random_state=42
)

# Generate subset for class 1 with 2 clusters
X1, y1 = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=2, n_classes=2, weights=[0.0, 1.0],
    class_sep=1.5, flip_y=0, random_state=24
)

# Combine datasets
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape labels
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Network architecture
input_dim = 2
hidden_dim = 4
output_dim = 1

# Parameter initialization
np.random.seed(1)
W1 = np.random.randn(hidden_dim, input_dim) * 0.01
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.randn(output_dim, hidden_dim) * 0.01
b2 = np.zeros((output_dim, 1))

# Training setup
eta = 0.1
epochs = 200
train_losses = []

# Training loop
for epoch in range(epochs):
    # Forward Pass
    Z1 = X_train.dot(W1.T) + b1.T
    A1 = sigmoid(Z1)

    Z2 = A1.dot(W2.T) + b2.T
    A2 = sigmoid(Z2)

    # Loss Calculation
    loss = binary_cross_entropy(y_train, A2)
    train_losses.append(loss)

    # Backward Pass
    dZ2 = bce_derivative(y_train, A2) * sigmoid_derivative(Z2)
    dW2 = dZ2.T.dot(A1) / X_train.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = dZ1.T.dot(X_train) / X_train.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Parameter Update
    W1 -= eta * dW1
    b1 -= eta * db1.T
    W2 -= eta * dW2
    b2 -= eta * db2.T

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation on test set
Z1 = X_test.dot(W1.T) + b1.T
A1 = sigmoid(Z1)
Z2 = A1.dot(W2.T) + b2.T
A2 = sigmoid(Z2)

y_pred = (A2 > 0.5).astype(int)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot training loss
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
#save image
plt.savefig("docs/exercise3/ex2_training_loss.png")
plt.show()

# Decision boundary visualization
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]

Z1 = grid.dot(W1.T) + b1.T
A1 = sigmoid(Z1)
Z2 = A1.dot(W2.T) + b2.T
A2 = sigmoid(Z2)
preds = (A2 > 0.5).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test.ravel(), edgecolor='k', cmap=plt.cm.Paired)
plt.title("Decision Boundary (Test Set)")

#save image
plt.savefig("docs/exercise3/ex2_decision_boundary.png")
plt.show()
