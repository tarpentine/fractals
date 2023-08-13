import torch
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Triangle wave function - distance from x to nearest integer
def triangle_wave(x):
    return torch.abs(x - torch.round(x))

def blancmange(x, n):
    result = 0
    for k in range(n):
        result += triangle_wave(torch.pow(torch.tensor(2.0), torch.tensor(k)) * x) * torch.pow(torch.tensor(0.5), torch.tensor(k))
    return result

size = 10000
n = 20
x = torch.linspace(0, 1, size).to(device)
y = blancmange(x, n)
plt.plot(x, y)
plt.axis('equal')
plt.show()
