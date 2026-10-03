import torch
weight = torch.tensor(
    2.0,
    requires_grad=True
)
input_value = torch.tensor(3.0)
target = torch.tensor(10.0)
learning_rate = 0.01

optimizer = torch.optim.SGD(
    [weight],
    lr=learning_rate
)
for step in range(10):
    prediction = weight * input_value
    loss = (prediction - target) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"Step {step + 1}: "
        f"Weight = {weight.item():.4f}, "
        f"Prediction = {prediction.item():.4f}, "
        f"Loss = {loss.item():.4f}"
    )