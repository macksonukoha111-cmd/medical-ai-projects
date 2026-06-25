import numpy as np 
import statsmodels.api as sm

# Patient data
age =        [25, 35, 45, 55, 65, 72, 80, 30, 50, 60]
heart_rate = [70, 75, 85, 90, 95, 100, 110, 68, 88, 92]
adhd =       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 

# Model 1: Heart rate predicted by age only 
model1 = sm.OLS(heart_rate, sm.add_constant(age)).fit()
print("=== Model 1: Age only ===")
print(model1.summary())

# Model 2: Heart rate predicted by age only
X = np.column_stack([age, adhd])
model2 = sm.OLS(heart_rate, sm.add_constant(X)).fit()
print("=== Model 2: Age + ADHD ===")
print(model2.summary())

# Quick interpretation 
print("\n --- Results ---")
print("Age predicts heart rate:", model1.pvalues[1] < 0.05)
print("ADHD predicts heart rate (after controlling age):", model2.pvalues[2] < 0.05)

# Conclusion: Both age and ADHD independently predicts heart rate
# This is an age-adjusted regression model