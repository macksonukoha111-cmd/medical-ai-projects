import torch 
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

train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size

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

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)

print("Total records:")
print(len(dataset))

print("\nTraining records:")
print(len(train_dataset))

print("\nTesting reords:")
print(len(test_dataset))

print("\nTraining batches:")

for features, labels in train_loader:
    print("Features:")
    print(features)
    print("Features shape:")
    print(features.shape)
    print("Labels:")
    print(labels)
    print("Labels shape:")
    print(labels.shape)
    print("------------------------------")