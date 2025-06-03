import torch

def get_data_sin(x_start, x_end):
    x = torch.linspace(x_start, x_end, 5000)
    y = 0.01 * x + 0.02 + 0.035 * torch.sin(x * 2)
    return torch.stack([x, y], dim=1)

def get_data_sin_rand(x_start, x_end):
    data = get_data_sin(x_start, x_end)
    x = data[:, 0]
    y = data[:, 1]
    y = y + (0.001 + 0.0015 * x) * torch.exp(torch.randn_like(x))
    return torch.stack([x, y], dim=1)