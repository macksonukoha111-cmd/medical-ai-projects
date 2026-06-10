import numpy as np

print("=== CREATING A STIMULATED BRAIN SCAN ===\n")

# Create a fake brain scan; 64 pixels tall, 64 pixels wide, 30 slices deep
# Values range from 0 (black) to 100 (white)
brain_scan = np.random.randint(0, 100,size=(64, 64, 30))

print(f"Brain scan shape: {brain_scan.shape}")
print(f"Total voxels (3D pixels): {brain_scan.size}")
print(f"avearge brightness: {np.mean(brain_scan):.2f}")
print(f"Min value: {brain_scan.min()}, Max value; {brain_scan.max()}")

print("\n=== EXTRACTING SLICES (LIKE VIEWING ONE MRI SLICE) ===\n")

# Get the middle slice (slice number 15 out of 0-29)
middle_slice = brain_scan[:, :, 15]
print(f"Middle slice shape: {middle_slice.shape}")

# Get the first 10 slices
first_10 = brain_scan[:, :, :10]
print(f"First 10 slices shape: {first_10.shape}")

# Get slices 10 through 20
slices_10_to_20 = brain_scan[:, :, 10:21]
print(f"Slices 10-20 shape: {slices_10_to_20.shape}")

print("\n=== NORMALIZING (MAKING VALUES 0-1)===\n")

# Before normalization
print(f"Before - Min: {brain_scan.min()}, Max:{brain_scan.max()}")

# Normalize to 0-1 range
normalized = (brain_scan - brain_scan.min()) / (brain_scan.max() - brain_scan.min())

print(f"After - Min: {normalized.min():.2f}, Max: {normalized.max():.2f}")
print("\n Day 1 complete!")