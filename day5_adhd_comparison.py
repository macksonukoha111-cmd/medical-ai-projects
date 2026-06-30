import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("===DAY 5: ADHD vs CONTROL GROUP COMPARISON ===\n")

# Load the patient data
df = pd.read_csv('patients.csv')
print("First 5 patients:")
print(df.head())

# Check if 'condition' column exista
# If not, let's simulate one for learning
if 'condition' not in df.columns:
    print("\n No 'condition' column found. Creating simulated data. . . ")
    # Create a condition columns: 0 = control, 1 = ADHD
    np.random.seed(42)  # For reproducibility
    df['condition'] = np.random.choice([0,1], size=len(df) , p=[0.5, 0.5])
    df['condition_label'] = df['condition'].map({0: 'Control', 1:'ADHD'})
else:
    df['condition_label'] = df['condition'].map({0: 'Control', 1:'ADHD'})

# Split the data
adhd = df[df['condition'] == 1]
control = df[df['condition'] == 0]

print(f"\n=== GROUP SIZES ===")
print(f"ADHD patients: {len(adhd)}")
print(f"Control pattients : {len(control)}")

print("\n=== VITAL SIGNS COMPARISON ===")

# Compare ages
adhd_age_mean = np.mean(adhd['age'])
control_age_mean = np.mean(control['age'])
print(f"ADHD average age: {adhd_age_mean:.1f} years ")
print(f"CONTROL average age: {control_age_mean:.1f} years")

# Compare heart rates
adhd_hr_mean = np.mean(adhd['heart_rate'])
control_hr_mean = np.mean(control['heart_rate'])
print(f"ADHD average heart rate: {adhd_hr_mean:.1f} bpm")
print(f"CONTROL average heart rate: {control_hr_mean:.1f} bpm")

# Compare blood pressure
adhd_bp_mean = np.mean(adhd['blood_pressure'])
control_bp_mean = np.mean(control['blood_pressure'])
print(f"ADHD average blood pressure: {adhd_bp_mean:.1f} mmHg")
print(f"CONTROL average blood pressure: {control_bp_mean:.1} mmHg")

# MY PERSONAL ASSIGNMENT TO CALCULATE THE DIFFERENCE BETWEEN ADHD AND CONTROL[AGE, BP, HR]
age_diff = adhd_age_mean - control_age_mean 
print(f" Age difference (ADHD - CONTROL): {age_diff:.1f} years")

hr_diff = adhd_hr_mean - control_hr_mean
print(f"Heart rate difference (ADHD - CONTROL): {hr_diff:.1f} bpm")

bp_diff = adhd_bp_mean - control_bp_mean
print(f"Blood pressure difference (ADHD - CONTROL): {bp_diff:.1f} mmHg")


# VISUALIZATION
print("\n=== VISUALIZATION ===")

# I will create bar chart for heart rates
plt.figure(figsize=(10, 6))
groups = ['Control''ADHD']
hr_values = [control_hr_mean, adhd_hr_mean]
plt.bar(groups, hr_values, color=['blue','red'])
plt.xlabel('Group')
plt.ylabel('Average Heart Rate (bpm)')
plt.title('Heart Rate: ADHD vs CONTROL')
plt.ylim(50, 100)
plt.grid(True, alpha=0.3)
plt.savefig('adhd_hr_comparison.png')
print("Heart rate comparison chart saved as 'adhd_hr_comparison.png'")

# I will create bar chart for blood pressure
plt.figure(figsize=(10, 6))
groups = ['Control', 'ADHD']
bp_values = [control_hr_mean, adhd_hr_mean]
plt.bar(groups, bp_values, color=['blue','red'])
plt.xlabel('Group')
plt.ylabel('Average Blood Rate (mmHg)')
plt.title('Blood Pressure: ADHD vs CONTROL')
plt.ylim(100, 150)
plt.grid(True, alpha=0.3)
plt.savefig('adhd_bp_comparison.png')
print("Blood pressure comparison chart saved as 'adhd_bp_comparison.png'")
print("\n Day 5 completed!")