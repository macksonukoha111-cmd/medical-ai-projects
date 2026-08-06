import pandas as pd
import numpy as np

# ___ Patient dataset with multiple clinical dimensions _____
data = {
    'patient_id':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'gender':      ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'department':  ['Cardiology', 'Cardiology', 'Neurology', 'Neurology',
                    'Cardiology', 'Neurology', 'Cardiology', 'Neurology',
                    'Cardiology', 'Cardiology', 'Neurology',  'Neurology'],
    'risk_level':  ['Low', 'high', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low'],
    'length_of_stay': [2, 7, 3, 5, 8, 2, 4, 9, 1, 5, 6, 2],
    'cost':         [1200, 5400, 1800, 3200, 6100, 1400, 2800, 7200, 900, 3400, 4900, 1300]                
}
df = pd.DataFrame(data)
print("=" * 50)
print("PATIENT DATASET:")
print("=" * 50)
print(df)

# ___ Step 1: Multiple aggregation at once_______________________
print("\n=== Step 1: Multiple stats per group ===")
dept_stats = df.groupby('department')['cost'].agg(['mean', 'count', 'std', 'min', 'max'])
print(dept_stats)

# ___ Step 2: Multiple columns, multiple aggrevations ____________________
print("\n=== Step 2: Multiple columns aggrevated ===")
multi_stats = df.groupby('department').agg({
    'cost': ['mean', 'sum'],
    'length_of_stay': ['mean', 'max']
})
print(multi_stats)

# __ Step 3: Group by TWO columns at once___________
print("\n=== Step 3: Group by department AND risk level ===")
two_group = df.groupby(['department', 'risk_level'])['cost'].mean()
print(two_group)

# ___ Step 4: pivot table - thr Excel-style summary grid ______________
print("\n=== Step 4: Pivot table ===")
pivot = pd.pivot_table(
    df,
    values='cost',
    columns='risk_level',
    aggfunc='mean'
)
print(pivot)

# __ Step 5: Pivot table with counts (how many patients per cell) -
print("\n=== Step 5: Patient count per department/risk combo ===")
pivot_count = pd.pivot_table(
    df,
    values='patient_id',
    index='department',
    columns='risk_level',
    aggfunc='count',
    fill_value=0
)
print(pivot_count)

# ____ Step 6: Highest-cost department/risk combination _________
print("\n=== Step 6: Clinical insight ===")
highest = pivot.stack().idxmax()
print(f"Highest average cost combination: {highest[0]} + {highest[1]} risk")
print(f"Average cost: ${pivot.stack().max():.2f}")