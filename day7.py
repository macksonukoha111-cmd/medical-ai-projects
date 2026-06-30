import pandas as pd
import matplotlib.pyplot as plt

# Patients dataset
data = {
    'age':             [25, 35, 45, 55, 65, 72, 80, 30, 50, 60],     
    'heart_rate':      [70, 75, 85, 90, 95, 100, 10, 68, 88, 92],
    'blood_pressure':  [110, 118, 125, 135, 145, 155, 160, 112, 130, 140],
    'risk_score':      [0, 0, 1, 1, 2, 2, 3, 0, 1, 2] 
}

df = pd.DataFrame(data)

# Calculate correlation matrix
corr = df.corr()
print("Correlation Matrix:")
print(corr)

# Visualize as heatmap
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', vmin=1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Patient Data Correlation Heatmap")
plt.tight_layout()
plt.savefig("day7_correlation_heatmap.png")
print("saved!")