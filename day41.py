import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.DataFrame({
    "age": [22, 25, 31, 35, 40, 44, 48, 52, 56, 60, 24, 28, 33, 38, 43, 47, 51, 55, 59, 63],
    "heart_rate": [72, 75, 78, 80, 82, 85, 88, 90, 94, 98, 76, 79, 81, 84, 87, 91, 95, 99, 102, 110],
    "risk": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,]
})

X = data[["age", "heart_rate"]]
y = data["risk"]

logistic_model = LogisticRegression(
    class_weight="balanced"
)

forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X,y)
forest_model.fit(X,y)

logistic_coefficients = pd.DataFrame({
    "feature": X.columns,
    "coefficient": logistic_model.coef_[0]
})

forest_importance = pd.DataFrame({
    "feature": X.columns,
    "Importance":
forest_model.feature_importances_})

print("Logistic Regression coefficients:")
print(logistic_coefficients)

print("\nRandom Forest feature importance:")
print(forest_importance)