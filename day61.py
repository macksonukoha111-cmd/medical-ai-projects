import torch
import torch.nn as nn

torch.manual_seed(42)

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

# Create one synthetic grayscale image
image = torch.rand(1, 1, 64, 64)

print("Input image shape:")
print(image.shape)

# Create the model
model = SimpleCNN()

print("\nModel:")
print(model)

# Pass the image through the feature extraction layers
features = model.features(image)

print("\nFeatures shape:")
print(features.shape)

# Pass the features through the classifier
prediction = model(image)

print("\nPrediction shape:")
print(prediction.shape)

print("\nPrediction:")
print(prediction)

print("\nNumber of trainable parameters:")
print(sum(parameter.numel() for parameter in model.parameters()))