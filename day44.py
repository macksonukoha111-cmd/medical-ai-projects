import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "age": [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    "group": [
        "A", "B", "A", "B", "A",
        "B", "A", "B", "A", "B",
        "A", "B", "A", "B", "A",
        "B", "A", "B", "A", "B"
    ]
})

X = data[["age", "heart_rate"]]
y = data["risk"]
groups = data["group"]

X_train, X_test, y_train, y_test, groups_train, group_test = train_test_split(
    X,
    y,
    groups,
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
model_pipeline.predict(X_test)

results = X_test.copy()
results["actual_risk"] = y_test.to_numpy()
results["predicted_risk"] = predictions
results["group"] = group_test.to_numpy()

print("Overall accuracy:")
print(accuracy_score(y_test, predictions))

print("\nOverall confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nDetailed results:")
print(results.sort_index())

for group_name in sorted(results["group"].unique()):
    group_results = results[results["group"] == group_name]
    actual_group = group_results["actual_risk"]
    predicted_group = group_results["predicted_risk"]
    print("\n--------------------------------------------------------------------")
    print("Group:", group_name)
    print("Number of records:", len(group_results))
    print("Accuracy:", accuracy_score(actual_group, predicted_group))
    print("Recall:", recall_score(
        actual_group,
        predicted_group,
        zero_division=0
        ))
    print("Confusion matrix:")
    print(confusion_matrix(
        actual_group,
        predicted_group,
        labels=[0,1]
        ))