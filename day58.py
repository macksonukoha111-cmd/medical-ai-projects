import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Create one synthethic grayscale image
image = torch.rand(64, 64)

print("Original image shape:")
print(image.shape)

# Add a channel dimension
image_with_channel = image.unsqueeze(0)

print("\nImage with channel shape:")
print(image_with_channel.shape)

# Add a batch dimension
image_batch = image_with_channel.unsqueeze(0)

print("\nImage batch shape:")
print(image_batch.shape)

print("\nNumber of images:")
print(image_batch.shape[0])

print("\nNumber of images:")
print(image_batch.shape[1])

print("\nImage heights:")
print(image_batch.shape[2])

print("\nImage width:")
print(image_batch.shape[3])

# Display the image
plt.imshow(
    image_batch[0, 0],
    cmap="gray"
)

plt.title("Grayscale Image from a 40 Tensor")
plt.xlabel("Width")
plt.ylabel("Height")
plt.colorbar(label="Pixel intensity")

plt.savefig(
    "day58_image_dimensions.png"
)

plt.show()
print("\nSaved diagram:")
print("day58_image_dimensions.png")