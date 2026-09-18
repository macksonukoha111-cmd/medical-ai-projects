import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import resample

data = pd.DataFrame({
    "age":        [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk":       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
})
print("Original class distribution:")
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

training_data = pd.concat([X_train, y_train], axis=1)
low_risk_train = training_data[training_data["risk"] == 0]
high_risk_train = training_data[training_data["risk"] == 1]

high_risk_oversampled= resample(
    high_risk_train,
    replace=True,
    n_samples=len(low_risk_train),
    random_state=42
)

balanced_training_data = pd.concat([low_risk_train, high_risk_oversampled])

print("\nTraining distribution before oversampling:")
print(training_data["risk"].value_counts())

print("\nTraining distribution after oversampling:")
print(balanced_training_data["risk"].value_counts())

X_train_balanced = balanced_training_data[["age", "heart_rate"]]
y_train_balanced = balanced_training_data["risk"]

model = LogisticRegression()
model.fit(X_train_balanced, y_train_balanced)

predictions = model.predict(X_test)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

































