# Writing patient data to a file
file = open("patients.txt", "w")
file.write("John, 120, Too high\n")
file.write("Ada, 55, Too low\n")
file.write("Chidi, 80, Normal\n")
file.close()

print("Patient data saved!")

# Reading patient data from file
print("--- Reading patient file ---")
file = open("patients.txt", "r")
for line in file:
    print(line)
file.close()

#Using Pandas 
import pandas as pd 

data = {
    "name": ["John", "Ada", "Chidi"],
    "heart_rate": [120, 55, 80],
    "age": [45, 32, 28]
}

df = pd.DataFrame(data)
print(df)

# Create a CSV file
import pandas as pd 

data = {
    "name": ["John", "Ada", "Chidi", "Ngozi", "Emeka"],
    "age": [45, 32, 28, 55, 41],
    "heart_rate": [120, 55, 80, 95, 110],
    "blood_pressure": [140, 110, 120, 135, 150]
}

df = pd.DataFrame(data)
df.to_csv("patients.csv", index=False)
print("CSV file created!")
print(df)