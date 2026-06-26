import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt 

# The three age groups and their blood pressure
young =  [110, 112, 108, 115, 111, 109, 113, 110, 114, 108]
middle_aged = [125, 128, 130, 122, 127, 132, 126, 129, 124, 131]
elderly  =  [150, 155, 148, 160, 152, 157, 145, 158, 153, 149]

# Run ANOVA
f_stat, p_value = stats.f_oneway(young, middle_aged, elderly)
print(f"P-value: {p_value:.4f}")

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value; {p_value:.4f}")

if p_value < 0.05:
    print("Result: Significant difference across age groups")
else:
    print("Result: no significant difference")

# Visualize
plt.figure(figsize=(8, 5))
plt.boxplot([young, middle_aged, elderly], labels=['Young', 'Middle_aged', 'Elderly'])
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure Across Age Groups (ANOVA)")
plt.savefig("day9_anova_boxplot.png")
print("Saved!")
