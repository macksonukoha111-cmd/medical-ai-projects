import torch

scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Scaler:")
print(scalar)

print("\nVector:")
print(vector)

print("\nMatrix:")
print(matrix)

print("\nVector shape:")
print(vector.shape)

print("\nMatrix shape:")
print(matrix.shape)

print("\nMatrix data type:")
print(matrix.dtype)