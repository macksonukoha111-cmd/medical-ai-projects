import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
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

logistic_model = LogisticRegression(
    class_weight="balanced"
)

forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

logistic_auc = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cross_validator,
    scoring="roc_auc"
)

forest_auc = cross_val_score(
    forest_model,
    X,
    y,
    cv=cross_validator,
    scoring="roc_auc"
)

print("Logistic Regression ROC-AUC scores:")
print(logistic_auc)

print("\nRandom Forest ROC-AUC scores:")
print(forest_auc)

print("\nLogistic Regression average ROC-AUC:")
print(logistic_auc.mean())

print("\nRandom Forest average ROC-AUC:")
print(forest_auc.mean())
