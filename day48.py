import torch

weight = torch.tensor(
    2.0,
    requires_grad=True
)

input_value = torch.tensor(3.0)
target = torch.tensor(10.0)
prediction = weight * input_value
loss = (prediction - target) ** 2
loss.backward()

print("Weight:")
print(weight)

print("\nPrediction:")
print(prediction)

print("\nLoss:")
print(loss)

print("\nGradient of weight:")
print(weight.grad)
