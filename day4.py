import pandas as pd
# Load the CSV file
df = pd.read_csv("patients.csv")

# Basic aalysis
print("Average heart rate:", df["heart_rate"].mean())
print("Average age:", df["age"].mean())
print("Highest blood presure:", df["blood_pressure"].max())
print("Lowest heart rate:", df["heart_rate"].min())

# Filter high risk patients
high_risk = df[df["heart_rate"] > 100]
print("\nHigh risk patients:")
print(high_risk)
