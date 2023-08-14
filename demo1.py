import torch
import numpy as np
import matplotlib.pyplot as plt
print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.tensor(X)
y = torch.tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
# z = torch.exp(-(x**2+y**2)/2.0)
# Compute sine
# z = torch.sin(x + y)
# Compute both, multiplied:
z = torch.sin(x + y) * (torch.exp(-(x**2+y**2)/2.0))

#plot
plt.imshow(z.numpy())
plt.tight_layout()
plt.show()
