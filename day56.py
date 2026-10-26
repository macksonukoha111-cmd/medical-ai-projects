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
validation_size = 2

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False
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

training_losses = []
validation_losses = []


for epoch in range(epochs):
    model.train()
    total_training_loss = 0.0

    for features, labels in train_loader:
        predictions = model(features)

        loss = loss_function(predictions, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_training_loss += loss.item()

    average_training_loss = total_training_loss / len(train_loader)
    training_losses.append(average_training_loss)
    model.eval()
    total_validation_loss = 0.0

    with torch.no_grad():
        for features, labels in validation_loader:
            predictions = model(features)
            loss = loss_function(
                predictions,
                labels
            )

            total_validation_loss += loss.item()

    average_validation_loss = (
            total_validation_loss / len(validation_loader)
            )
    validation_losses.append(average_validation_loss)

    print(
            f"Epoch {epoch + 1}: "
            f"Average loss = {average_training_loss:.4f}"
            f"Validation loss = {average_validation_loss:.4f}"
            )

print("Training loss values:",len(training_losses))
print("Validation loss values:",len(validation_losses))
plt.plot(
    range(1, epochs + 1),
    training_losses,
    marker="o",
    label="Training loss"
    )
plt.plot(
    range(1, epochs + 1),
    validation_losses,
    marker="s",
    label="Validation loss"
)
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("day56_training_validation_loss.png")
print("Graph saved successfully")
plt.show()
