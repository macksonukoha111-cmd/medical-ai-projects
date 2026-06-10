from sklearn.tree import DecisionTreeClassifier
import numpy as np 

# Training data - heart rates and whether high risk (1) or not (0)
heart_rates = [[120], [55], [80], [95], [110], [70], [45], [130]]
labels =      [1,      1,     0,   0,     1,     0,    0,     1]

# Train the model
model = DecisionTreeClassifier()
model.fit(heart_rates, labels)


# Test with new patients
test = [[175], [115], [50]]
predictions = model.predict(test)

for i, rate in enumerate(test):
    risk = "High Risk" if predictions[i] == 1 else "Nomal"
    print(f"Heart rate {rate[0]} -> {risk}")
