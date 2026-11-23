import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

torch.manual_seed(42)

# Create one synthetic grayscale image
image = torch.rand(1, 1, 64, 64)

print("Input image shape:")
print(image.shape)

# Create a convolutional layer
convolution = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3
)

# Create ReLU activation
relu = nn.ReLU()

# Create max-pooling layer
pooling = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

# Apply convolution
feature_maps = convolution(image)

print("\nAfter convolution:")
print(feature_maps.shape)

# Apply ReLU
activated_features = relu(feature_maps)

print("\nAfter ReLU:")
print(activated_features.shape)

# Apply max pooling
pooled_features = pooling(activated_features)

print("\nAfter max pooling:")
print(pooled_features.shape)

# Select thee first image and first feature map
first_feature_map = activated_features[0, 0]
first_pooled_map = pooled_features[0, 0]

print("\nFirst feature map shape:")
print(first_feature_map.shape)

print("\nFirst pooled map shape:")
print(first_pooled_map.shape)

# Display the feature map and pooled map
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(
    first_feature_map.detach(),
    cmap="gray"
)
plt.title("After ReLU")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "day60_relu_max_pooling.png"
)
plt.show()
print("\nSaved diagram:")
print("day60_relu_max_pooling.png")
