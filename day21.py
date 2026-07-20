import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#____________Create messy medical dataset____________
data = {
    'patient_id':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'age':          [25, None, 45, 55, 200, 72, 80, 30, None, 60],
    'blood_pressure':[110, 118, None, 135, 145, 999, 160, 112, 130, 140],
    'heart_rate':    [70, 75, 85, 90, 95, 100, None, 68, 88, 92],
    'gender':        ['M', 'F', 'M', 'f', 'M', 'F', 'm', 'F', 'M', None],
    'risk_score':    [0, 0, 1, 1, 2, 2, 3, 0, 1, 2],
    'notes':         ['normal', 'normal', 'elevated', 'elevated', 'ELEVATED', 'high', 'HIGH', 'normal', 'elevated', 'high']
}

df = pd.DataFrame(data)

print("=" * 50)
print("ORIGINAL MESSY DATA:")
print("=" * 50)
print(df)
print(f"\nShape: {df.shape}")
print(f"\nMisssing vakues:\n{df.isnull().sum()}")

#______ Step 1: Fix impossible values______________
print("\n=== Step 1: Remove impossible values ===")
df.loc[df['age'] > 120, 'age'] = np.nan
df.loc[df['blood_pressure'] > 300, 'blood_pressure'] = np.nan
print("Impossible values replaced with NaN")

#______Step 2: Fill missing values _________
print("\n=== Step 2: fill missing values ===")
df['age'].fillna(df['age'].median(), inplace=True)
df['blood_pressure'].fillna(
    df['heart_rate'].median(), inplace=True)
df['heart_rate'].fillna(
    df['heart_rate'].median(), inplace=True)
print(f"Missing values after fill:\n{df.isnull().sum()}")

# _____ Step 3: Standardize text columns _________
print("\n=== Step 3: Standardize text ===")
df['gender'] = df['gender'].str.upper()
df['gender'].fillna('UNKNOWN', inplace=True)
df['notes'] = df['notes'].str.lower()
print("Gender and notes standardized")

# ____ Step 4: Remove duplicates______
df = df.drop_duplicates()
print(f"\n=== Step 4: Shape after removing duplicates ===")
print(f"Shape: {df.shape}")

#____ Step 5: Final clean dataset______________________
print("\n=== CLEAN DATASET ===")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

#____ Visualize before and after__________
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Data Cleaning - Age Distribution', fontweight='bold')

# Before - with outlier
before_age = [25, np.nan, 45, 55, 200, 72, 80, 30, np.nan, 60]
before_age_clean = [x for x in before_age
                    if x is not None and not np.isnan(x)]
axes[0].hist(before_age_clean, bins=6,
             color='tomato', edgecolor='black')
axes[0].set_title('Before Cleaning\n(200 is impossible outlier)')
axes[0].set_xlabel('Age')

# After
axes[1].hist(df['age'], bins=6,
              color='steelblue', edgecolor='black')
axes[1].set_title('After Cleaning\n(outlier removed)')
axes[1].set_xlabel('Age')

plt.tight_layout()
plt.savefig("day21_data_cleaning.png", dpi=150)
print("\nSaved!")