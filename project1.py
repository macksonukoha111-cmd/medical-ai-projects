import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Patiemt data
data = {
    "name": ["John", "Ada", "Chidi", "Ngozi", "Emeka"],
     "age": [45, 32, 28, 55, 41],
     "heart_rate": [120, 55, 80, 95, 110],
     "blood_pressure": [140, 110, 120, 135, 150]
}

df = pd.DataFrame(data)
print("=== PATIENT RISK ASSESMENT SYSTEM ===")
print(df)
print("\nAverage heart rate:", df["heart_rate"].mean())
print("High risk patients:")
print(df[df["heart_rate"] > 100][["name", "heart_rate"]])

import matplotlib.pyplot as plt

# Add risk column
df["risk"] = df["heart_rate"].apply(
    lambda x: "High Risk"if x > 100 else "Normal"  
)

print("\nFull patient assessment:")
print(df[["name", "heart_rate", "blood_pressure", "risk"]])

# Save chart
colors = []
for r in df["risk"]:
    if r == "High Risk":
        colors.append("red")
    else:
        colors.append("green")

plt.figure()
plt.bar(df["name"], df["heart_rate"], color=colors)        
plt.title("Patient Risk Assessment")
plt.xlabel("Patient")
plt.ylabel("Heart Rate")
plt.savefig("risk_assessment.png")
print("\nChart saved!")
