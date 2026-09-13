import torch
from swe_constants import *

def topography(x, y):
    return torch.zeros_like(x)

def topography_grad(x, y):
    # Частные производные z_x, z_y
    return torch.zeros_like(x), torch.zeros_like(y)

def residual(model, x, y, t):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    h, u, v = model(x, y, t)

    # --- производные по времени ---
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    hu_t = torch.autograd.grad(h * u, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    hv_t = torch.autograd.grad(h * v, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    # --- производные по x ---
    hu_x = torch.autograd.grad(h * u, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    hu2_x = torch.autograd.grad(h * u * u, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    huv_x = torch.autograd.grad(h * u * v, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h2_x = torch.autograd.grad(h * h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    # --- производные по y ---
    hv_y = torch.autograd.grad(h * v, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    huv_y = torch.autograd.grad(h * u * v, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    hv2_y = torch.autograd.grad(h * v * v, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h2_y = torch.autograd.grad(h * h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    z_x, z_y = topography_grad(x, y)
    z = topography(x, y)

    # Невязки 3 уравнений

    R1 = h_t + hu_x + hv_y

    R2 = hu_t + hu2_x + 0.5 * g * h2_x + huv_y + g * h * z_x

    R3 = hv_t + huv_x + hv2_y + 0.5 * g * h2_x + g * h * z_y

    # Возвращает (N, 3), каждый R размера (N, 1)
    return torch.cat([R1, R2, R3], dim=1)


