import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ____ Dataset __ more patients _____
X_train = np.array([
    [0.2, 0.3, 0.1],
    [0.5, 0.8, 0.6],
    [0.9, 0.95, 0.85],
    [0.1, 0.2, 0.15],
    [0.7, 0.85, 0.75],
    [0.3, 0.4, 0.25],
])
y_train = np.array([[0], [1], [1], [0], [1], [0]])

# New patients the model has never seen
X_test = np.array([
    [0.4, 0.6, 0.4],
    [0.8, 0.9, 0.8],
    [0.15, 0.25, 0.1],
])
y_test = np.array([[1], [1], [0]])

#___ Training function with L2 regularizaation____
def train_network(X, y, learning_rate=0.5, epochs=2000, lambda_reg=0.01):

    np.random.seed(42)
    W1 = np.random.randn(3, 4) * 0.1
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.1
    b2 = np.zeros((1, 1))

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = X @ W1 + b1
        A1 = sigmoid(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)

        # Training loss with 12 penalty
        mse_loss = np.mean((y - A2) ** 2)
        L2_penalty = (lambda_reg / 2) * (np.sum(W1**2) + np.sum(W2**2))
        train_loss = mse_loss + L2_penalty
        train_losses.append(train_loss)

        # Test loss (no regularization for evaluation)
        Z1_test = X_test @ W1 + b1
        A1_test = sigmoid(Z1_test)
        Z2_test = A1_test @ W2 + b2
        A2_test = sigmoid(Z2_test)
        test_loss = np.mean((y_test - A2_test) ** 2)
        test_losses.append(test_loss)

        # Backward pass
        dA2 = -(y - A2)
        dZ2 = dA2 * sigmoid_derivative(Z2)
        dW2 = A1.T @ dZ2 + lambda_reg * W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * sigmoid_derivative(Z1)
        dW1 = X.T @ dZ1 + lambda_reg * W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 400 == 0:
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Test loss={test_loss:.4f}")

    return W1, b1, W2, b2, train_losses, test_losses
# ___ Train ____________________
print("=== Training with L2 Regularization ===")
W1, b1, W2, b2, train_losses, test_losses = train_network(X_train, y_train, lambda_reg=0.01)

print("\n--- Test Patient Predictions ---")
Z1 = X_test @ W1 + b1
A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

for i, (pred, true) in enumerate(zip(A2, y_test)):
    label = "HIGH RISK" if pred > 0.5 else "LOW RISK"
    correct = "CORRECT" if (pred > 0.5) == (true == 1) else "WRONG"
    print(f"Test Patient {i+1}: " f"{pred[0]:.2%} -> {label} - {correct}")

#___ Plot train vs test loss ______
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='steelblue', linewidth=2)
plt.plot(test_losses, label='Test loss', color='tomato', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Test Loss - Checking for Overfitting")
plt.legend()
plt.grid(True)
plt.savefig("day19_regularization.png", dpi=150)
print("\nSaved!")


