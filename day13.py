import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

data = {
    'age':             [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],
    'heart_rate':      [70, 75, 85, 90, 95, 100, 110, 68, 88, 92],
    'blood_pressure':  [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'adhd':            ['ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD'],
    'risk_score':      [0, 0, 1, 1, 2, 2, 3, 0, 1, 2] 
}
df = pd.DataFrame(data)

sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Medical Data Analysis - Seaborn', fontsize=16, fontweight='bold')

# Plot 1: Regression plot
sns.regplot(data=df, x='age', y='blood_pressure', ax=axes[0,0], color='steelblue')
axes[0,0].set_title("Age vs Blood Pressure (with regression)")

# Plot 2: Violin plot
sns.violinplot(data=df, x='adhd', y='blood_pressure', ax=axes[0,1], palette='Set2')
axes[0,1].set_title("Blood Pressure Distribution by ADHD")

# Plot 3: Heatmap
corr = df[['age', 'heart_rate', 'blood_pressure', 'risk_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1,0], fmt='.2f')
axes[1,0].set_title("Correlation Heatmap")

# Plot 4 - Scatter with hue
sns.scatterplot(data=df, x='age', y='heart_rate', hue='adhd', size='risk_score', sizes=(50, 200), ax=axes[1, 1])
axes[1,1].set_title("Age vs Heart Rate by ADHD & Risk")

plt.tight_layout()
plt.savefig("day13_seaborn.png", dpi=150)
print("Saved!")