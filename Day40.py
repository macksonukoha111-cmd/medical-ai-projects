import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

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

model = LogisticRegression(
    class_weight="balanced"
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[: , 1]

brier_score = brier_score_loss(y_test, probabilities)

roc_auc = roc_auc_score(y_test, probabilities)

print("Actual test labels:")
print(y_test.to_numpy())

print("\nPredicted high-risk probabilities:")
print(probabilities)

print("\nBrier score:")
print(brier_score)

print("\nROC-AUC:")
print(roc_auc)