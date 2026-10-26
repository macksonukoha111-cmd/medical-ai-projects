import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

patient_data = torch.tensor([
    [22.0, 72.0],
    [40.0, 82.0],
    [60.0, 98.0],
    [35.0, 76.0],
    [52.0, 90.0],
    [28.0, 68.0],
    [67.0, 105.0],
    [45.0, 85.0]
])

risk_labels = torch.tensor([
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    1.0,
    1.0
])

dataset = TensorDataset(
    patient_data,
    risk_labels
)

train_size = 6
test_size = 2

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=True
)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.00001
)

epochs = 20
loss_history = []

for epoch in range(epochs):
    total_loss = 0.0

    for features, labels in train_loader:
        predictions = model(features)

        loss = loss_function(predictions, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"Average loss = {average_loss:.4f}"
        )

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
    )
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig("day55_training_loss.png")
plt.show()

print("\nFirst training loss:")
print(f"{loss_history[0]:.4f}")

print("\nFinal training loss:")
print(f"{loss_history[-1]:.4f}")

print("\nSaved diagram:")
print("day55_training_loss.png")