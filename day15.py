import numpy as np

#____ 1. Creating arrays_________
vector = np.array([1, 2, 3, 4, 5])
print("Vector:", vector)
print("Sahpe:", vector.shape)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMatrix:\n", matrix)
print("Shape:", matrix.shape)

#___2. Special arrays__________
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
random_matrix = np.random.rand(2, 3)

print("\nZeros:\n", zeros)
print("\nIdentity:\n", identity)
print("\nRandom:\n", random_matrix)

# ___ 3. Element-wise opertions_____
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nAddition:", a + b)
print("Multiplication:", a * b)
print("Squared:", a ** 2)

# ___4. Matrix multiplication {the core of neural networks} ___
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5 ,6],
              [7, 8]])

elementwise = A * B
dot_product = A @ B          # or np.dot(A, B)

print("\nElement-wise multiply:\n", elementwise)
print("\nMatrix multipication (dot product):\n", dot_product)

# __5. Reshaping__________
arr = np.arange(12)
print("\nOriginal:", arr)
reshaped = arr.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped)

#___6. Transpose_____
print("\nTransposed:\n", reshaped.T)