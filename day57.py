import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

image = torch.rand(62, 64)

print("Image tensor:")
print(image)

print("\nImage shape:")
print(image.shape)

print("\nImage data type:")
print(image.dtype)

print("\nMinimum pixel value:")
print(image.min().item())

print("\nMaximum pixel value:")
print(image.max().item())

print("\nAverage pixel value:")
print(image.mean().item())

plt.imshow(
    image,
    cmap="gray"
)
plt.title("Synthetic Medical Image Tensor")
plt.xlabel("Width")
plt.ylabel("Height")
plt.colorbar(label="Pixel intensity")

plt.savefig("day57_synthetic_image.png")
plt.show()
print("\nSaved diagram:")
print("day57_synthetic_image.png")