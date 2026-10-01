import torch

patient_data = torch.tensor([
    [22.0, 72.0],
    [40.0, 82.0],
    [60.0, 98.0]
])

print("Patient data:")
print(patient_data)

print("\nData type:")
print(patient_data.dtype)

print("\nData shape:")
print(patient_data.shape)

age = patient_data[:, 0]
heart_rate =  patient_data[:, 1]

print("\nAges:")
print(age)

print("\nHeart rates:")
print(heart_rate)

print("\nAge plus 10 years:")
print(age + 10)

print("\nHeart rate multiplied by 2:")
print(heart_rate * 2)

print("\nAverage age:")
print(torch.mean(age))

print("\nReshaped patient data:")
print(patient_data.reshape(2, 3))
