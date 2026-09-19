import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.DataFrame({
    "age":        [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk":       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
})

print("Class distribution:")
print(data["risk"].value_counts())

X = data[["age", "heart_rate"]]
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))