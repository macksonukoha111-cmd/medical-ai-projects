import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm 
import matplotlib.pyplot as plt

# _____ Dataset ___________________________________
data = {
    'age':  [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],
    'heart_rate':  [70, 75, 85, 90, 95, 100, 110, 68, 88, 92],
    'blood_pressure':  [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'adhd':  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'risk_score': [0, 0, 1, 1, 2, 2, 3, 0, 1, 2]
}
df = pd.DataFrame(data)

# __1. Correlation__________________
print("=== CORRELATION ===")
print(df.corr())

# __2. T-Test: ADHD vs Non ADHD blood pressure __________
print("\n==T-TEST==")
adhd_bp =  df[df['adhd'] == 1]['blood_pressure'].values
no_adhd_bp = df[df['adhd'] == 0]['blood_pressure'].values
t_stat, p_val = stats.ttest_ind(adhd_bp, no_adhd_bp)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value; {p_val:.4f}")

#__3. Regression: Age predicts blood pressure__________
print("\n ===REGRESSION===")
model = sm.OLS(df['blood_pressure'], sm.add_constant(df['age'])).fit()
print(f"R-squared: {model.rsquared:.4f}")
print(f"Age p_value: {model.pvalues['age']:.4f}")

# __4. Visualization____________________
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1 - Scatter
axes[0].scatter(df['age'], df['blood_pressure'])
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Blood Pressure")
axes[0].set_title("Age vs Blood Pressure")

# Plot 2 - Boxplot
axes[1].boxplot([adhd_bp, no_adhd_bp], tick_labels=['ADHD', 'No ADHD'])
axes[1].set_ylabel("Blood Pressure")
axes[1].set_title("ADHD vs No ADHD")

# Plot 3 - Risk score bar chart
axes[2].bar(df['age'], df['risk_score'])
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Risk Score")
axes[2].set_title("Risk Score by Age")

plt.tight_layout()
plt.savefig("day10_full_analysis.png")
print("\nSaved!")