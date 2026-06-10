import numpy as np

print("=== DAY 2: WHAT SHAPES REALLY MEAN ===\n")

# Differences ways to represent the same medical data
# 4 patients, 3 vital signs: [age, heart_rate,bp]

# Shape (4, 3) - 4 rows, 3 columns 
patients = np.array([
    [25, 72, 118],
    [45, 78, 135],
    [60, 82, 145],
    [35, 70, 122]
])
print("Shape (4, 3):")
print(patients)
print(f"Shape: {patients.shape}\n")

# shape (3, 4) - 3 rows, 4 columns (transposed)
patients_T = patients.T
print("Shape (3, 4) - transposed:")
print(patients_T)
print(f"Shape: {patients_T.shape}\n")

# What does ecah shape mean?
print("=== MEDICAL INTERPRETATION ===\n")
print("Shape (4, 3): 4 patients, each row has 3 measurements")
print("Shape (3, 4): 3 measurements, each rowhas4 patients")

print("\n=== WHY THIS MATTERS ===\n")
print("Neural networks expect: (batch_size, features)")
print("So 4 patients, 3 features -> shapes (4, 3) is correct")
print("Shape (3, 4) would be wrong and cause errors")

# YOUR TURN
print("\n=== YOUR CHALLENGE ===\n")

# Create data for 10 adhd patients, each with 5 brain measurements
adhd_patients = np.random.randint(50, 150, size=(10, 5))
print(f"ADHD data shape: {adhd_patients.shape}")
print("This means: 10 patients, 5 meausrements per patient")

# Calculate average of each measurement across all patients
averages = np.mean(adhd_patients, axis=0)
print(f"Average of each measurement: {averages}")

# Calculate average for each patient (acrossss thier 5 measurements)
patient_average = np.mean(adhd_patients, axis=1)
print(f"Each patient`s averages: {patient_average}")