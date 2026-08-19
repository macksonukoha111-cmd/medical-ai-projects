import pandas as pd
import numpy as np

# ____ Table 1: Patient demographics (from registration system) _____
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6],
    'name':       ['Alice', 'Bob', 'Carla', 'David', 'Emma', 'Frank'],
    'age':        [25, 40, 45, 55, 62, 30],
    'gender':     ['F', 'M', 'F', 'M', 'F', 'M']
})

# ____ Table 2: Lab results (from lab system) _________________
# Notice: patient 6 has no lab results, patient 7 doesn't exist in patients
labs = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6],
    'blood_pressure': [110, 135, 145, 160, 130, 150],
    'cholesterol': [180, 220, 190, 240, 200, 210]
})

print("=" * 50)
print("PATIENTS TABLE:")
print(patients)

print("\n" + "=" * 50)
print("LABS TABLE:")
print("=" * 50)
print(labs)

# ____ Step 1: Inner join __ only patients present in BOTH tables ____
print("\n=== Step 1: INNER JOIN ===")
inner = pd.merge(patients, labs, on='patient_id', how='left')
print(inner)
print(f"Shape: {inner.shape}")

# ____ Step 2: Left join - keep All patients, even without labs _____
print("\n=== Step 2: LEFT JOIN ===")
left = pd.merge(patients, labs, on='patient_id', how='left')
print(left)
print(f"Shape: {left.shape}")

# ___ Step 3: Outer join - keep EVERYTHING from both tables ____
print("\n=== Step 3: OUTER JOIN ===")
outer = pd.merge(patients, labs, on='patient_id', how='outer')
print(outer)
print(f"Shape: {outer.shape}")


# ____ Step 4: Fill missing labs after the left join ________
print("\n=== Step 4: Handle missing lab data ===")
left['blood_pressure'].fillna(left['blood_pressure'].median(), inplace=True)
print(left)

# ___ Step 5: Groupby - average cholesterol by gender ____
print("\n=== Step 5: Groupby analysis ===")
summary = left.groupby('gender')[['blood_pressure', 'cholesterol']].mean()
print(summary)