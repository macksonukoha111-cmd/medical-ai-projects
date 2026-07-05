import numpy as np
import matplotlib.pyplot as plt

# ___ Activation functions ____
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# __ Patient data ___-
X = np.array([

    [0.2, 0.3, 0.1],
    [0.5, 0.8, 0.6],
    [0.9, 0.95, 0.85],
    [0.1, 0.2, 0.15],
    [0.7, 0.85, 0.75],
    [0.3, 0.4, 0.25],
])

y = np.array([[0], [1], [1], [0], [1], [0]])

# ___ Initialize weights for two layers____
np.random.seed(42)

# Layer 1: 3 inputs -> 4 hidden neurons
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros((1, 4))

# Layer 2: 4 hidden neurons -> 1 output
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))

learning_rate = 0.5
losses = []

# ____ Training loop ___
for epoch in range(2000):

    # __Forward pass___
    # Layer 1
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    # Layer 2 
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # Loss
    loss = np.mean((y - A2) ** 2)
    losses.append(loss)

    # __ Backward Pass___
    # Output layer gradients
    dA2 = -(y - A2)
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # ___ Update weights________
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# ___  Final predictions________
print("\n--- Final predictions ---")
for i, (pred, true) in enumerate(zip(A2, y)):
    label = "HIGH RISK" if pred > 0.5 else "LOW RISK"
    correct = "CORRECT" if (pred > 0.5) == (true == 1) else "WRONG"
    print(f"Patient {i+1}: {pred[0]:.2%} -> {label} - {correct}")

# ____ Plot __________________
plt.figure(figsize=(8, 5))
plt.plot(losses, color='tomato', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multi-Layer Neural Network - Learning Curve")
plt.grid(True)
plt.savefig("day18_nn_learning_curve.png", dpi=150)
print("\nSaved!")