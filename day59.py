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

# Pass the image through the convolutional layer
feature_maps = convolution(image)

print("\nFeature maps shape:")
print(feature_maps.shape)

print("\nNumber of output features maps:")
print(feature_maps.shape[1])

print("\nFeature map height:")
print(feature_maps.shape[2])

print("\nFeatures map width:")
print(feature_maps.shape[3])

# Select the first image and first features map
first_feature_map = feature_maps[0, 0]

print("\nFirst feature map shape:")
print(first_feature_map.shape)

# Display the first feature map
plt.imshow(
    first_feature_map.detach(),
    cmap="gray"
)

plt.title("First Convolutional Feature Map")
plt.xlabel("Width")
plt.ylabel("Height")
plt.colorbar(label="Activation")

plt.savefig(
    "day59_feature_map.png"
)

plt.show()

print("\nSaved diagram:")
print("day59_feature_map.png")