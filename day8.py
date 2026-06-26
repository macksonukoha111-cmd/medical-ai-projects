import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 

# Two groups 
adhd_bp = [135, 140, 145, 150, 138, 142, 148, 155, 137, 143]
no_adhd_bp = [110, 115, 118, 112, 120,108, 14, 116, 119, 111]

# Run T-Test
t_stat, p_value = stats.ttest_ind(adhd_bp, no_adhd_bp)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Significant difference between groups")
else:
    print("Result: No significant difference")

# Visualize
plt.figure(figsize=(8, 5))
plt.boxplot([adhd_bp, no_adhd_bp], labels=['ADHD', 'No ADHD'])
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure: ADHD vs No ADHD")
plt.savefig("day8_ttest_boxplot.png")
print("Saved!")    

# ADHD patients showed significantly higher blood pressure compared to non ADHD patients.