import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Softmax activation
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy(y, y_hat):
    eps = 1e-9
    return -np.mean(np.sum(y * np.log(y_hat + eps), axis=1))

# Derivative of cross-entropy wrt logits (softmax combined)
def cross_entropy_derivative(y, y_hat):
    return (y_hat - y)

# One-hot encoding
def one_hot(y, num_classes):
    one_hot_matrix = np.zeros((y.size, num_classes))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix

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

# Combine
X = np.vstack((X0, X1, X2))
y = np.hstack((y0, y1, y2))

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert labels to one-hot
num_classes = 3
y_train_oh = one_hot(y_train, num_classes)
y_test_oh = one_hot(y_test, num_classes)

# Network architecture
input_dim = 4
hidden1_dim = 8
hidden2_dim = 6
output_dim = 3

# Initialize parameters
np.random.seed(1)
b1 = np.zeros((hidden1_dim, 1))
b2 = np.zeros((hidden2_dim, 1))
b3 = np.zeros((output_dim, 1))

W1 = np.random.randn(hidden1_dim, input_dim) / np.sqrt(input_dim)
W2 = np.random.randn(hidden2_dim, hidden1_dim) / np.sqrt(hidden1_dim)
W3 = np.random.randn(output_dim, hidden2_dim) / np.sqrt(hidden2_dim)

# Training setup
eta = 0.3
epochs = 300
train_losses = []

# Training loop
for epoch in range(epochs):
    # Forward Pass
    Z1 = X_train.dot(W1.T) + b1.T
    A1 = np.tanh(Z1)

    Z2 = A1.dot(W2.T) + b2.T
    A2 = np.tanh(Z2)

    Z3 = A2.dot(W3.T) + b3.T
    A3 = softmax(Z3)

    # Loss Calculation
    loss = cross_entropy(y_train_oh, A3)
    train_losses.append(loss)

    # Backward Pass
    dZ3 = cross_entropy_derivative(y_train_oh, A3)
    dW3 = dZ3.T.dot(A2)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3.dot(W3)
    dZ2 = dA2 * (1 - np.tanh(Z2) ** 2)
    dW2 = dZ2.T.dot(A1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2)
    dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)
    dW1 = dZ1.T.dot(X_train)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Normalize gradients
    dW3 /= X_train.shape[0]
    db3 /= X_train.shape[0]
    dW2 /= X_train.shape[0]
    db2 /= X_train.shape[0]
    dW1 /= X_train.shape[0]
    db1 /= X_train.shape[0]

    # Parameter Update
    W1 -= eta * dW1
    b1 -= eta * db1.T
    W2 -= eta * dW2
    b2 -= eta * db2.T
    W3 -= eta * dW3
    b3 -= eta * db3.T

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation
Z1 = X_test.dot(W1.T) + b1.T
A1 = np.tanh(Z1)
Z2 = A1.dot(W2.T) + b2.T
A2 = np.tanh(Z2)
Z3 = A2.dot(W3.T) + b3.T
A3 = softmax(Z3)

y_pred = np.argmax(A3, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot training loss
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


#save image
plt.savefig("docs/exercise3/ex4_loss.png")
plt.show()