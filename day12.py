import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'age':            [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],  
    'heart_rate':     [70, 75, 85, 90, 95, 100, 110, 68, 88, 92], 
    'blood_pressure': [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'adhd':           [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
    'risk_score':     [0, 0, 1, 1, 2, 2, 3, 0, 1, 2] 
}
df = pd.DataFrame(data)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Advanced Clinical Analysis', fontsize=16, fontweight='bold')

#________Plot 1: Scatter with trend line and groups_______
colors = ['steelblue' if x == 0 else 'tomato' for x in df['adhd']]
axes[0].scatter(df['age'], df['blood_pressure'], c=colors, s=120, edgecolors='black', zorder=3)

# Reference - hypertension threshold
axes[0].axhline(y=140, color='red', linestyle='--', linewidth=1.5, label='Hypertension (140)')

# Annotation
axes[0].annotate('Hypertension\nThreshold', xy=(26, 141), fontsize=9, color='red')

axes[0].set_ylabel("Age")
axes[0].set_xlabel("Blood Pressure")
axes[0].set_title("BP by Age & ADHD Status\n(Blue=No ADHD, Red=ADHD)")

# ___ Plot 2: Heart rates zones _______________
zone_colors = []
for hr in df['heart_rate']:
    if hr < 75:
        zone_colors.append('green')
    elif hr < 95:
        zone_colors.append('orange')
    else:
        zone_colors.append('red')

bars = axes[1].bar(df['age'], df['heart_rate'], color=zone_colors, edgecolor='black')

axes[1].axhline(y=75, color='green', linestyle='--', linewidth=1)
axes[1].axhline(y=95, color='orange', linestyle='--', linewidth=1)
axes[1].set_ylabel("Age")
axes[1].set_xlabel("Heart Rate")
axes[1].set_title("Heart Rate Zones by Age\n(Green=Normal, Orange=Elevated, Red=High")

plt.tight_layout()
plt.savefig("day12_advanced_plots.png", dpi=150)
print("Saved!")