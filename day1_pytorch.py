import torch

# Three patients' heart rates
heart_rates = torch.tensor([72.0, 85.0, 95.0])

# Print the tensor
print(heart_rates)

# Print the data type
print(heart_rates.dtype)

# Calculate the mean
mean_hr = heart_rates.mean()
print(mean_hr)


import torch

# Group A: treated patients 
group_a = torch.tensor([72.0, 85.0, 95.0])

# Group B: controlled patients
group_b = torch.tensor([68.0, 72.0, 85.0])

# Calculate the both mean
group_a = group_a.mean()
group_b = group_b.mean()

# Print the mean
print(group_a)
print(group_b)

# Calculate the differences between both mean
diff = group_a - group_b
print(diff.item())

import torch

# Group A : treated patients 
group_a = torch.tensor([72.0, 85.0, 95.0])

# Group B : controlled patients
group_b = torch.tensor ([68.0, 75.0, 82.0])

# Group C :different patients
group_c = torch.tensor ([80.0, 88.0, 92.0])

# Calculate the three mean
mean_a = group_a.mean()
mean_b = group_b.mean()
mean_c = group_c.mean()

# Print the three mean
print(mean_a)
print(mean_b)
print(mean_c)

# The differences between the three mean
diff = mean_a - mean_b - mean_c
print(diff.item())


# Find the best treatment mean
best = max(mean_c.item(), mean_b.item(), mean_a.item())
print("The best treatment mean:", best)


# Compare treatment C and control treatment
vs_control = mean_c.item() - mean_b.item()
print("Treatment C beats Control treatment by:", vs_control, "bpm")