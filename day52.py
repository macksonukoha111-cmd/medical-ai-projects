import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

patient_data = torch.tensor([
    [22.0, 72.0],
    [40.0, 82.0],
    [60.0, 98.0]
])

predictions = model(patient_data)

print("Model:")
print(model)

print("\nPatient data:")
print(patient_data)

print("\nPatient data shape:")
print(patient_data.shape)

print("\npedictions:")
print(predictions)

print("\nPrediction shape:")
print(predictions.shape)

print("\nTrainable parameters:")

for name, parameter in model.named_parameters():
    print(name, parameter.shape)