import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.DataFrame({
    "age": [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,]
})

X = data[["age", "heart_rate"]]
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced"))
])

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)

results = X_test.copy()
results["actual_risk"] = y_test
results["predicted_risk"] = predictions

results["error_type"] = "Correct"

results.loc[
    (results["actual_risk"] == 0) &
    (results["predicted_risk"] == 1),
    "error_type"
] = "False Positive"

results.loc[
    (results["actual_risk"] == 1) &
    (results["predicted_risk"] == 0),
    "error_type"
] = "False Negative"

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nDetailed prediction results:")
print(results.sort_index())
