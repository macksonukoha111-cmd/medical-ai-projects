import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#-----------------------------
# 1. Create a synthetic dataset
#------------------------------

number_of_images = 100
image_size = 64

images = torch.rand(
    number_of_images,
    1,
    image_size,
    image_size
) * 0.2

labels = torch.zeros(
    number_of_images,
    dtype=torch.long
)

# Add a bright square to half of the images
for index in range(number_of_images // 2, number_of_images):
    images[index, 0, 24:40, 24:40] = 1.0
    labels[index] = 1

print("Images shape:")
print(images.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nClass 0 records:")
print((labels == 0).sum().item())

print("\nClass 1 records:")
print((labels == 1).sum().item())

shuffled_indices = torch.randperm(number_of_images)
shuffled_images = images[shuffled_indices]
shuffled_labels = labels[shuffled_indices]

training_images = shuffled_images[:80]
training_labels = shuffled_labels[:80]

testing_images = shuffled_images[80:]
testing_labels = shuffled_labels[80:]

print("\nTraining images shape:")
print(training_images.shape)

print("\nTesting images shape:")
print(testing_images.shape)

print("\nTraining Class 0 records:")
print((training_labels == 0).sum().item())

print("Training Class 1 records:")
print((training_labels == 1).sum().item())

print("Testing Class 0 records:")
print((testing_labels == 0).sum().item())

print("Testing Class 1 records:")
print((testing_labels == 1).sum().item())

# ---------------------------------
# 3. Define the CNN
# ---------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=4 * 31 * 31,
                out_features=2
            )
        )
    def forward(self, image):
        features = self.features(image)
        output = self.classifier(features)
        return output

model = SimpleCNN()
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

number_of_epochs = 30

for epoch in range(number_of_epochs):
    model.train()
    predictions = model(training_images)
    loss = loss_function(
        predictions,
        training_labels
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    accuracy = (
        predicted_classes == training_labels
    ).float().mean()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d} |"
            f"Loss: {loss.item():.4f} |"
            f"Accuracy: {accuracy.item():.2%}"
        )

model.eval()
with torch.no_grad():
    testing_predictions = model(testing_images)
    testing_classes = torch.argmax(
        testing_predictions,
        dim=1
    )

    testing_accuracy = (testing_classes == testing_labels).float().mean()

print("\nFinal testing accuracy:")
print(f"{testing_accuracy.item():.2%}")

print("\nActual testing labels:")
print(testing_labels)

print("\nPredicted testing labels:")
print(testing_classes)