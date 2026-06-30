import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

data = {
    'age':  [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],
    'heart_rate': [70, 75, 85, 90, 95, 100, 110, 68, 88, 92],
    'blood_pressure': [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'adhd': ['ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD', 'ADHD', 'No ADHD'],
    'risk_score': [0, 0, 1, 1, 2, 2, 3, 0, 1, 2]
}
df = pd.DataFrame(data)

# Professional medical color palette
sns.set_theme(style='whitegrid', palette='muted')

# GridSpec for custom layout
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Patient Cardiovascular Risk Report', fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')

# _____ Plot 1: Large regression plot(top, spans 2 columns) ___
ax1 = fig.add_subplot(gs[0, :2])
sns.regplot(data=df, x='age', y='blood_pressure', ax=ax1, color='#2980b9', scatter_kws={'s': 100, 'edgecolors': 'black'}, line_kws={'linewidth': 2})
ax1.axhline(y=140, color='red', linestyle='--', linewidth=1.5)
ax1.text(26, 141, 'Hypertension Threshold', color='red', fontsize=9)
ax1.set_title('Blood Pressure vs Age', fontweight='bold')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Blood Pressure (mmhg)')

# ______ Plot 2: Risk distribution (top right)_________
ax2 = fig.add_subplot(gs[0, 2])
risk_counts = df['risk_score'].value_counts().sort_index()
colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
ax2.pie(risk_counts.values, labels=[f'Risk {i}' for i in risk_counts.index], colors=colors[:len(risk_counts)], autopct='%1.1f%%', startangle=90)
ax2.set_title('Risk Distribution', fontweight='bold')

# _____ Plot 3: Violin plot (middle left)_________
ax3 = fig.add_subplot(gs[1, :2])
sns.violinplot(data=df, x='adhd', y='blood_pressure', ax=ax3, palette=['#3498db', '#e74c3c'])
ax3.set_title('Blood Pressure by ADHD Status', fontweight='bold')
ax3.set_xlabel('Group')
ax3.set_ylabel('Blood Pressure (mmHg)')

# ____ Plot 4: Heatmap (middle right)_____
ax4 = fig.add_subplot(gs[1, 2])
corr = df[['age', 'heart_rate', 'blood_pressure', 'risk_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4, fmt='.2f', linewidth=0.5)
ax4.set_title('Correlation', fontweight='bold')

# ____ Plot 5: Heart rate trend (bottom, full width)_______
ax5 = fig.add_subplot(gs[2, :])
colors_hr = []
for hr in df['heart_rate']:
    if hr < 75:
        colors_hr.append('green')
    elif hr < 95:
        colors_hr.append('orange')
    else:
        colors_hr.append('red')        
ax5.set_xticks(range(len(df)))
ax5.set_xticklabels([f'Age {a}' for a in df['age']], rotation=45)
ax5.axhline(y=75, color='green', linestyle='--', linewidth=1)
ax5.axhline(y=95, color='orange', linestyle='--', linewidth=1)
ax5.set_title('Heart Rate by Patients Age', fontweight='bold')
ax5.set_ylabel('Heart Rate (bpm)')

plt.savefig("day14_medical_report.png", dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
print("Medical report saved!")