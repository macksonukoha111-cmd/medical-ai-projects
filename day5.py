import pandas as pd
import matplotlib.pyplot as plt

# load patient data 
df = pd.read_csv("patients.csv")

# Create a bar chart of heart rates 
plt.bar(df["name"], df["heart_rate"], color="red")
plt.title("Patient Heart Rates")
plt.xlabel("Patient Name")
plt.ylabel("Heart Rate")
plt.savefig("heart_rates.png")
print("Chart saved as heart_rates.png")

# Blood pressure chart
plt.figure()
plt.bar(df["name"], df["blood_pressure"], color="blue")
plt.title("Patient Blood Pressure")
plt.xlabel("Patient Name")
plt.ylabel("Blood Pressure")
plt.savefig("blood_pressure.png")
print("Blood pressure chart saved!")