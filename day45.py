import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.DataFrame({
    "age": [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,]
})

X = data[["age", "heart_rate"]]
y = data["risk"]

cross_validator = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced"))
])

forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model
}

metrics = {
    "Recall": "recall",
    "Balanced Accuracy": "balanced_accuracy",
    "ROC-AUC": "roc_auc",
    "Average Precision": "average_precision"
}

results = []
for model_name, model in models.items():
    row = {"Model":model_name}

    for metric_name, scoring_name in metrics.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cross_validator,
            scoring=scoring_name
        )

        row[metric_name] = scores.mean()
        results.append(row)
        results_table = pd.DataFrame(results)

        print("Final model comparison:")
        print(results_table.round(3))