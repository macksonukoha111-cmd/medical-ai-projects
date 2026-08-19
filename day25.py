import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("DAY 25 CAPSTONE: Cardiology & Neurology Patient Analysis")
print("=" * 60)

# ── 1. LOAD (simulated multi-source data) ───────────────────────
demographics = pd.DataFrame({
    'patient_id': range(1, 16),
    'name': ['Alice','Bob','Carla','David','Emma','Frank','Grace',
             'Henry','Ivy','Jack','Kelly','Liam','Mona','Noah','Olivia'],
    'age': [25, 40, None, 55, 62, 30, 45, 200, 38, 50, 29, 61, 33, None, 47],
    'gender': ['F','M','F','M','F','M','f','M','F','M','F','M','F','F','M'],
    'department': ['Cardiology','Cardiology','Neurology','Neurology',
                    'Cardiology','Neurology','Cardiology','Neurology',
                    'Cardiology','Cardiology','Neurology','Neurology',
                    'Cardiology','Neurology','Cardiology']
})

labs = pd.DataFrame({
    'patient_id': [1,2,3,4,5,6,7,8,9,10,11,12,16],  # 16 doesn't exist in demographics
    'blood_pressure': [110, 135, 999, 160, 145, 130, 118, 155, None, 142, 128, 165, 120],
    'cholesterol': [180, 220, 190, 240, 200, 175, 210, 260, 195, 230, 185, 250, 205]
})

outcomes = pd.DataFrame({
    'patient_id': [1,2,3,4,5,7,8,9,10,11,13,14,15],  # 6, 12 missing on purpose
    'length_of_stay': [2, 7, 3, 5, 8, 4, 9, 1, 5, 6, 3, 2, 4],
    'cost': [1200, 5400, 1800, 3200, 6100, 2800, 7200, 900, 3400, 4900, 1600, 1100, 2600],
    'risk_level': ['Low','High','Low','Medium','High','Medium','High',
                    'Low','Medium','High','Low','Low','Medium']
})

print(f"\nDemographics: {len(demographics)} | Labs: {len(labs)} | Outcomes: {len(outcomes)}")

# ── 2. CLEAN ─────────────────────────────────────────────────────
print("\n=== Cleaning ===")
demographics.loc[demographics['age'] > 120, 'age'] = np.nan
demographics['age'].fillna(demographics['age'].median(), inplace=True)
demographics['gender'] = demographics['gender'].str.upper()

labs.loc[labs['blood_pressure'] > 300, 'blood_pressure'] = np.nan
labs['blood_pressure'].fillna(labs['blood_pressure'].median(), inplace=True)

print("Impossible age (200) and blood pressure (999) fixed")
print(f"Missing values remaining:\n{demographics.isnull().sum().sum()} in demographics, "
      f"{labs.isnull().sum().sum()} in labs")

# ── 3. MERGE ─────────────────────────────────────────────────────
print("\n=== Merging (left join on demographics — keep every patient) ===")
df = demographics.merge(labs, on='patient_id', how='left') \
                  .merge(outcomes, on='patient_id', how='left')
print(f"Final merged shape: {df.shape}")
print(f"Patients missing outcomes data: {df['cost'].isna().sum()}")

# ── 4. DERIVED FEATURES ──────────────────────────────────────────
df['high_bp_flag'] = np.where(df['blood_pressure'] > 140, 'Elevated', 'Normal')
df['high_chol_flag'] = np.where(df['cholesterol'] > 200, 'High', 'Normal')
df['cost_per_day'] = (df['cost'] / df['length_of_stay']).round(2)

# ── 5. GROUPBY / PIVOT SUMMARY ────────────────────────────────────
print("\n=== Average cost by department ===")
print(df.groupby('department')['cost'].agg(['mean', 'count']))

print("\n=== Pivot: average cost by department & risk level ===")
pivot = pd.pivot_table(df, values='cost', index='department',
                        columns='risk_level', aggfunc='mean')
print(pivot)

print("\n=== Elevated BP patients by department ===")
print(df[df['high_bp_flag'] == 'Elevated'].groupby('department')['patient_id'].count())

# ── 6. KEY INSIGHTS ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
highest_combo = pivot.stack().idxmax()
print(f"1. Highest-cost combo: {highest_combo[0]} + {highest_combo[1]} risk "
      f"(${pivot.stack().max():.2f} avg)")

missing = df[df['cost'].isna()]
print(f"2. {len(missing)} patients have no outcomes data yet: "
      f"{missing['name'].tolist()}")

elevated_both = df[(df['high_bp_flag']=='Elevated') & (df['high_chol_flag']=='High')]
print(f"3. {len(elevated_both)} patients have BOTH elevated BP and cholesterol "
      f"— highest cardiovascular risk group")

# ── 7. VISUALIZE ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.groupby('department')['cost'].mean().plot(kind='bar', ax=axes[0],
    color=['#2c3e50','#3498db'], edgecolor='black')
axes[0].set_title('Average Cost by Department', fontweight='bold')
axes[0].set_ylabel('Cost ($)')

pivot.plot(kind='bar', ax=axes[1], edgecolor='black')
axes[1].set_title('Cost by Department & Risk Level', fontweight='bold')
axes[1].set_ylabel('Cost ($)')

plt.tight_layout()
plt.savefig('day25_capstone.png', dpdayi=150)
print("\nSaved: day25_capstone.png")
print("\nDay 25 Capstone complete!")