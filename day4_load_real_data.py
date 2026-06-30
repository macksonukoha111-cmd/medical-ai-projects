import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== DAY 4: LOADING REAL PATIENT DATA===\n")

# load the CSV file
df =  pd.read_csv('patients.csv')

print("First 5 patients:")
print(df.head())

print ("\n===DATA SHAPE ===\n")
print (f"Total patients: {df.shape[0]}")
print(f"Measurements per patient:{df.shape[1]}")

print("\n=== COLUMN NAMES ===\n")
print(df.columns.tolist())

print("\n=== BASIC STATISTICS ===\n")
print(df.describe())

# Convert to Numpy array
data_array = df.values
print(f"\nNumPy array shape:{data_array.shape}")

df = pd.read_csv('patients.csv')

# Extract columns directly
age = df['age'].values
heart_rates = df['heart_rate'].values
blood_pressure = df['blood_pressure'].values

print(f"Average age: {np.mean(age):.1f}")
print(f"Max heart rate: {np.max(heart_rates)}")
print(f"Min blood pressure: {np.min(blood_pressure)}")
print(f"Standard deviation of age:{np.std(age):.1f}")