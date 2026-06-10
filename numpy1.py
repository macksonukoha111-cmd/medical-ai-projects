import numpy as np

# NumPy arrays
brain_volumes = np.array([1200, 1350, 980, 1100, 1420])

print("Brain volumes:", brain_volumes)
print("Average volume:", np.mean(brain_volumes))
print("Largestbrain:", np.max(brain_volumes))
print("Smallest brain:", np.min(brain_volumes))
print("Standard deviation:", np.std(brain_volumes))

# ADHD vs Normal comparison
adhd_volumes = np.array([980, 1050, 920, 1000, 960])
normal_volumes = np.array([1200, 1350, 1280, 1420, 1300])

print("\nADHD average volume:", np.mean(adhd_volumes))
print("Normal average volume:", np.mean(normal_volumes))
print("Diffference:", np.mean(normal_volumes) - np.mean(adhd_volumes))


import matplotlib.pyplot as plt
groups = ["ADHD", "Normal"]
averages = [np.mean(adhd_volumes), np.mean(normal_volumes)]
            
plt.bar(groups, averages, color=["orange", "blue"])
plt.title("ADHD vs Normal Brain Volume")
plt.ylabel("Average Voume (ml)")
plt.savefig("adhd_comparison.png")
print("Chart saved!")            
            
