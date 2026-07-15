"""
Medical Risk Prediction System
===============================
Author: Ukoha Omaka
Built: Day 20 of 90-Day Healthcare AI Engineering Curriculum
Description: A complete neural network built from scratch using only numpy to predict cardiovascular risk in patients.
Demonstrates: forward pass, backpropagation, regularization, and clinical intergration.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#-------------------------------------------------------------------------------------
# 1. DATA
#-------------------------------------------------------------------------------------

# Features: [age, blood_pressure, heart_rate] - normalized 0 to 1
X_train = np.array([
    [0.2, 0.3, 0.1],
    [0.5, 0.8, 0.6],
    [0.9, 0.95, 0.85],
    [0.1, 0.2, 0.15],
    [0.7, 0.85, 0.75],
    [0.3, 0.4, 0.25],
    [0.6, 0.75, 0.65],
    [0.15, 0.25, 0.12],
])
y_train = np.array([[0],[1],[1],[0],[1],[0],[1],[0]])

X_test = np.array([
    [0.4, 0.6, 0.4],
    [0.8, 0.9, 0.8],
    [0.15, 0.25, 0.1],
    [0.55, 0.7, 0.6],
])
y_test = np.array([[1],[1],[0],[1]])

#----------------------------------------------------------
# 2. NEURAL NETWORK CLASS
#----------------------------------------------------------

class MedicalRiskNN:
    """
    Two-layer neural network for cardiovascular risk prediction.
    Architecture: Input(3) -> Hidden(6) -> Output(1)
    """
    def __init__(self, learning_rate=0.5, lambda_reg=0.01):
        self.lr = learning_rate
        self.lam = lambda_reg
        np.random.seed(42)
        self.W1 = np.random.randn(3, 6) * 0.1
        self.b1 = np.zeros((1, 6))
        self.W2 = np.random.randn(6, 1) * 0.1
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def loss(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        l2 = (self.lam/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return mse + l2
    
    def backward(self, X, y):
        dA2 = -(y - self.A2)
        dZ2 = dA2 * self.sigmoid_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 + self.lam * self.W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.sigmoid_deriv(self.Z1)
        dW1 = X.T @ dZ1 + self.lam * self.W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, X_test, y_test, epochs=3000):
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            train_losses.append(self.loss(y, y_pred))
            test_pred = self.forward(X_test)
            test_losses.append(
                np.mean((y_test - test_pred)**2))
            self.forward(X)
            self.backward(X, y)

            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_losses[-1]:.4f} | "
                      f"Test Loss: {test_losses[-1]:.4f}")
                
        return train_losses, test_losses
    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        return probs, (probs >= threshold).astype(int)
    
#----------------------------------------------------------------------
# 3. TRAIN
#----------------------------------------------------------------------

print("=" * 50)
print(" MEDICAL RISK PREDICTION SYSTEM")
print(" Neural Network from scratch - NumPy")
print("=" * 50)

model = MedicalRiskNN(learning_rate=0.5, lambda_reg=0.1)
train_losses, test_losses = model.train(X_train, y_train, X_test, y_test, epochs=3000)

# ----------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------- 

print("\n--- Training Set Results ---")
probs, preds = model.predict(X_train)
correct = 0
for i, (prob, pred, true) in enumerate(
    zip(probs, preds, y_train)):
  label = "HIGH RISK" if pred == 1 else "LOW RISK"
  status = "CORRECT" if pred == true else "WRONG"
  if pred == true:
      correct += 1
  print(f"Patient {i+1}: {prob[0]:.2%} -> {label} - {status}")

print(f"Training Accuracy: {correct}/{len(y_train)}")

print("\n--- Test Set Results ---")
probs_t, preds_t = model.predict(X_test)
correct_t = 0
for i, (prob, pred, true) in enumerate(
    zip(probs_t, preds_t, y_test)):
  label = "HIGH RISK" if pred == 1 else "LOW RISK"
  status = "CORRECT" if pred == true else "WRONG"
  if pred == true:
      correct_t += 1
  print(f"Test Patient {i+1}: "
        f"{prob[0]:.2%} -> {label} - {status}")

print(f"\nTest Accuracy: {correct_t}/{len(y_test)}")

# -------------------------------------------------------------
# 5. VISUALIZE
# -------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
fig.suptitle('Medical Risk Prediction System - Day 20 Capstone', fontsize=16, fontweight='bold', color='#2c3e50')

# Plot 1 - Learning curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(train_losses, color='steelblue', linewidth=2, label='Training Loss')
ax1.plot(test_losses, color='tomato', linewidth=2, label='Test Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Learning Curve - Training vs Test Loss")
ax1.legend()
ax1.grid(True)

# Plot 2 - Training predictions
ax2 = fig.add_subplot(gs[1, 0])
train_probs = model.forward(X_train).flatten()
colors = ['tomato' if y == 1 else 'steelblue'
          for y in y_train.flatten()]
ax2.bar(range(len(train_probs)), train_probs, color=colors, edgecolor='black')
ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
ax2.set_xlabel("Patient")
ax2.set_title("Training Predictions\n" "(Red=High Risk, Blue=Low Risk)")
ax2.set_ylim(0, 1)

# Plot 3 - Test predictions
ax3 = fig.add_subplot(gs[1, 1])
test_probs = model.forward(X_test).flatten()
colors_t = ['tomato' if y == 1 else 'steelblue'
            for y in y_test.flatten()]
ax3.bar(range(len(test_probs)), test_probs, color=colors_t, edgecolor='black')
ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
ax3.set_xlabel("Patient")
ax3.set_ylabel("Risk Probability")
ax3.set_title("Test Predictions\n" "(Red=High Risk, Blue=Low Risk)")
ax3.set_ylim(0, 1)

plt.savefig("day20_capstone.png", dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
print("\nCapstone dashboard saved!")
print("\n" + "="*50)
print(" NumPy Neural Network Phase Complete")
print(" Days 15-20 Done")
print("="*50)            
