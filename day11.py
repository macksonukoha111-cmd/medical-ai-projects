import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Patient dataset
data = {
    'age':              [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],
    'heart_rate':       [70, 75, 85, 90, 95, 100, 110, 68, 88, 92], 
    'blood_pressure':   [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'adhd':             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
    'risk_score':       [0, 0, 1, 1, 2, 2, 3, 0, 1, 2]
}
df = pd.DataFrame(data)

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cardiovascular Risk Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1 - Age vs Blood Pressure
axes[0,0].scatter(df['age'], df['blood_pressure'], color='steelblue', s=100, edgecolors='black')
axes[0,0].set_xlabel("Age")
axes[0,0].set_ylabel("Blood Pressure")
axes[0,0].set_title("Age vs Blood Pressure")

# Plot 2 - Risk Score Distribution
axes[0,1].hist(df['risk_score'], bins=4, color='tomato', edgecolor='black')
axes[0,1].set_xlabel("Risk Score")
axes[0,1].set_ylabel("Count")
axes[0,1].set_title("Risk Score Distribution")

# Plot 3 - Blood Pressure by ADHD
adhd_groups = [df[df['adhd']==0]['blood_pressure'].values,df[df['adhd']==1]['blood_pressure'].values]
axes[1,0].boxplot(adhd_groups, tick_labels=['No ADHD', 'ADHD'])
axes[1,0].set_ylabel("Blood Pressure")
axes[1,0].set_title("Blood Pressure by ADHD Status")

# Plot 4 - Age vs Risk Score
axes[1,1].bar(df['age'], df['risk_score'], color='mediumseagreen', edgecolor='black')
axes[1,1].set_xlabel("Age")
axes[1,1].set_ylabel("Risk Score")
axes[1,1].set_title("Risk Score by Age")

plt.tight_layout()
plt.savefig("day11_dashboard.png", dpi=150)
print("Dashboard saved!")