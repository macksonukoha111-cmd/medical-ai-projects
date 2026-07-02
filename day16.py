import numpy as np

# ___ 1. Define a single neuron_______
# Patient data: [age, blood_pressure, heart_rate]
inputs = np.array([0.5, 0.8, 0.3])  # normalized values

# Weights - how important each input is 
weights = np.array([0.4, 0.7, 0.2])

# Bias - background adjustment
bias = 0.1

# ___ 2. Forward pass (raw output)____
raw_output = np.dot(inputs, weights) + bias
print(f"Raw output (before activation): {raw_output:.4f}")

# ___ 3. Activation functions _____
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Apply each activation
sig_output = sigmoid(raw_output)
relu_output = relu(raw_output)
tanh_output = tanh(raw_output)

print(f"\nAfter sigmoid: {sig_output:.4f}")
print(f"After relu:  {relu_output:.4f}")
print(f"After tanh:  {tanh_output:.4f}")

# __ 4. What sigmoid medically_____
print(f"\n-- Clinical Interpretation ---")
print(f"Risk probablity: {sig_output:.2%}")
if sig_output > 0.5:
    print("Prediction: HIGH RISK")
else:
    print("Prediction: LOW RISK")

# ___ 5. Try different patients ____
print("\n--- Multiple Patients ---")
patients = np.array([
    [0.2, 0.3, 0.1],   # young, low BP, Low HR
    [0.5, 0.8, 0.6],   # middle, high BP, elevated HR
    [0.9, 0.95, 0.85], # elderly, very high BP, high HR
])

for i, patients in enumerate(patients):
    raw = np.dot(patients, weights) + bias 
    risk = sigmoid(raw)
    label = "HIGH RISK" if risk > 0.5 else "LOW RISK"
    print(f"Patient {i+1}: risk={risk:.2%} -> {label}")