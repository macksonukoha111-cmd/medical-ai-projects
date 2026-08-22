import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


patients = pd.DataFrame({
    'age': [25, 40, 55, 62, 30, 45, 38, 50, 29, 61, 33, 47],
    'blood_pressure': [110, 135, 160, 145, 130, 118, 155, 142, 128, 165, 120, 150],
    'cholestrol': [180, 220, 240, 200, 175, 210, 260, 195, 185, 250, 205, 230],
    'risk_label': [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
})

# Features: information used by the model
X = patients[['age', 'blood_pressure', 'cholestrol']]

# Target: the outcomes the model learns to predict
y = patients['risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


print('TOTAL RECORDS:', len(patients))
print('TRAINING RECORDS:', len(X_train))
print('TEST RECORDS:', len(X_test))
print('\nTRAINING LABEL COUNTS:')
print(y_train.value_counts().sort_index())
print('\nREST LABEL COUNTS:')
print(y_test.value_counts().sort_index())


# Create and train the baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict labels for data the model did not see during training
predictions = model.predict(X_test)

print('\nTEST PREDICTIONS:')
print(predictions)

print('\nACTUAL TEST LABELS:')
print(y_test.to_numpy())

accuracy = accuracy_score(y_test, predictions)
print(f'\nTEST ACCURACY: {accuracy:.2f}')

matrix = confusion_matrix(y_test, predictions)

print('\nCONFUSION MATREIX:')
print(matrix)

tn, fp, fn, tp = matrix.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'\nSENSITIVITY: {sensitivity:.2f}')
print(f'SPECIFICITY: {specificity:.2f}')