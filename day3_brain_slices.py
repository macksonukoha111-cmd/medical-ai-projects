import numpy as np

print("=== DAY 3: EXTRACTING BRAIN REGIONS ===\n")

# Simulated brain MRI: 100 slices, even 12x128 pixels
# In real MRI: slices index = position from front to back of the head
brain = np.random.randint(0, 100, size=(128,128,100))
print("Full brain shape: {brain.shape}")
print("(height, width, slices) - slices go from front to back")

print("\n=== ANATOMICAL LANDMARKS (from anatomy) ===\n")
print("- Slices 0-20: Frontal lobe (decision making, attention)")
print("-Slices 20-50: Parietal lobe (sensory)")
print("-Slices 50-80: Occipital lobe (vision)")
print("_Slices 80-100: Cerebellum (cordination)")

print("\n=== EXTRACTING REGIONS ===\n")

# Extract each regoin using slicing
Frontal = brain[:, :, 0:20]
Parietal = brain[:, :, 20:50]
Occipital = brain[:, :, 50:80]
Cerebellum = brain[:, :,80-100]

print(f"Frontal lobe shape: {Frontal.shape}")
print(f"Parietal lobe shape: {Parietal.shape}")
print(f"Occipital lobe shape: {Occipital.shape}")
print(f"Cerebellum lobe shape: {Cerebellum.shape}")

Prefrontal = brain[:, :, 0:6]
print(f"Prefrontal shape: {Prefrontal.shape}")

# To calculate the average activity of prefrontal and occipital
prefrontal_avg = np.mean(Prefrontal)
occipital_avg = np.mean(Occipital)

print(f"Prefrontal average activity: {prefrontal_avg:.2f}")
print(f"Occipital average activity: {occipital_avg:.2f}")

if prefrontal_avg > occipital_avg:
    print("\n PREFRONTAL HAS HIGHER ACTIVITY THAN OCCIPITAL")
else:
    print("\n PREFRONTAL HAS LOWER ACTIVITY")

# TODO: extractnthe parietal lobe and calculate its average
# ANSWERS:
Parietal = brain[:, :, 20:40]
print(f"Parietal shape: {Parietal.shape}")

parietal_avg = np.mean(Parietal)
print(f"Parietal average activity: {parietal_avg:.2f}")
