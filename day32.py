from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

# 0 = llow risk, 1 = high risk
actual_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

# A poor model predicts low risk for everyone
predicted_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

accuracy = accuracy_score(actual_labels, predicted_labels)
matrix = confusion_matrix(actual_labels, predicted_labels)

print('ACCURACY:', accuracy)
print('\nCONFUSION MATRIX:')
print(matrix)

precision = precision_score(actual_labels, predicted_labels, zero_division=0)
recall = recall_score(actual_labels, predicted_labels, zero_division=0)
balanced_accuracy = balanced_accuracy_score(actual_labels, predicted_labels)

print('\nPRECISION:', precision)
print('RECALL / SENSITIVITY:', recall)
print('SPECIFICITY: 1.0')
print('BALANCED ACCURACY:', balanced_accuracy)