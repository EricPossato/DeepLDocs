import numpy as np

# Activation function
def tanh(x):
    return np.tanh(x)

# Derivative of tanh
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Input vector
x = np.array([0.5, -0.2])

# Target output
y = 1.0

# Hidden layer weights and biases
W1 = np.array([[0.3, -0.1],
               [0.2,  0.4]])
b1 = np.array([0.1, -0.2])

# Output layer weights and bias
W2 = np.array([0.5, -0.3])
b2 = 0.2

# Learning rate
eta = 0.3

# Forward Pass
z1 = W1.dot(x) + b1
print(f"z1 = {z1}")

h1 = tanh(z1)
print(f"h1 = {h1}")

u2 = W2.dot(h1) + b2
print(f"u2 = {u2}")

y_hat = tanh(u2)
print(f"y_hat = {y_hat}")

# Loss Calculation
L = (y - y_hat)**2
print(f"Loss L = {L}")

# Backward Pass
dL_dyhat = 2 * (y_hat - y)
print(f"dL/dy_hat = {dL_dyhat}")

dL_du2 = dL_dyhat * tanh_derivative(u2)
print(f"dL/du2 = {dL_du2}")

dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
print(f"dL/dW2 = {dL_dW2}")
print(f"dL/db2 = {dL_db2}")

dL_dh1 = dL_du2 * W2
print(f"dL/dh1 = {dL_dh1}")

dL_dz1 = dL_dh1 * tanh_derivative(z1)
print(f"dL/dz1 = {dL_dz1}")

dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
print(f"dL/dW1 =\n{dL_dW1}")
print(f"dL/db1 = {dL_db1}")

# Parameter Update
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1

print("\nUpdated Parameters:")
print(f"W2 = {W2_new}")
print(f"b2 = {b2_new}")
print(f"W1 =\n{W1_new}")
print(f"b1 = {b1_new}")
