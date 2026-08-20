import pandas as pd
import numpy as np

# Synthetic educational healthcare data
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6],
    'age': [25, 40, None, 200, 62, 30],
    'department': ['Cardiology', 'Neurology', 'Cardiology', 'None', 'Cardiology', 'Neurology'],
    'blood_pressure': [120, 135, 999, 160, None, 130],
    'cost': [1200, 5400, 1800, 3200, None, 2800]
})

print('PATIENT DATA')
print(patients)

print('\nMISSING VALUES BY COLUMN')
print(patients.isnull().sum())

print('\nTOTAL MISSING VALUES')
print(patients.isnull().sum().sum())

print('\nINVALID AGE RECORDS')
invalid_age = patients[patients['age'] > 120]
print(invalid_age)

print('\nINVALID BLOOD PRESSURE RECORDS')
invalid_bp = patients[patients['blood_pressure'] > 300]
print(invalid_bp)

print('\nRECORD MISSING DEPARTMENT')
print(patients[patients['department'].isnull()])

print('\nDATA QUALITY SUMMARY')
print(f'Total missing values: {patients.isnull().sum().sum()}')
print(f'Invalid age records: {len(invalid_age)}')
print(f'invalid blood pressure records: {len(invalid_bp)}')