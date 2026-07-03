import numpy as np
import matplotlib.pyplot as plt 

# ___ Activation function & derivative _____
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 -sigmoid(x))

# __ Patient data _____________________
# Inputs: [age, blood_pressure, heart_rate] normalized
X = np.array([
    [0.2, 0.3, 0.1],     # Patient 1 - LOW risk
    [0.5, 0.8, 0.6],     # Patient 2 - HIGH risk
    [0.9, 0.95, 0.85],   # Patient 3 - HIGH risk
    [0.1, 0.2, 0.15],    # Patient 4 - LOW risk
])

# True labels: 0 = low risk, 1 = high risk
y = np.array([[0], [1], [1], [0]])

# ___ Initialize random weights___________
np.random.seed(42)
weights = np.random.randn(3, 1) * 0.1
bias = 0.0
learning_rate = 0.5

# ____ Training loop__________
losses = []

for epoch in range(1000):
    # Forward pass
    raw = X @ weights + bias 
    predictions = sigmoid(raw)
    
    # Calculate error (loss)
    loss = np.mean((y - predictions) ** 2)
    losses.append(loss)

    # Backward pass (gradients)
    error = y - predictions
    d_pred = sigmoid_derivative(raw)
    gradient = X.T @ (error * d_pred)
    bias_gradient = np.mean(error * d_pred)

    # Update weights
    weights += learning_rate * gradient
    bias += learning_rate * bias_gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# ____ Final predictions______
print("\n--- Final Predictions ---")
final_preds = sigmoid(X @ weights + bias)
for i, (pred, true) in enumerate(zip(final_preds, y)):
    label = "HIGH RISK" if pred > 0.5 else "LOW RISK"
    correct = "CORRECT" if (pred > 0.5) == (true == 1) else "x"
    print(f"Patient {i+1}: {pred[0]:.2%} -> {label} {correct}")

# ____ Plot learning curve__________
plt.figure(figsize=(8, 5))
plt.plot(losses, color='steelblue', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neuron Learning Curve - Loss Decreasing Over Time")
plt.grid(True)
plt.savefig("day17_learning_curve.png", dpi=150)
print("\nLearning curve saved!")            

