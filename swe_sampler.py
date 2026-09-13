import torch
from swe_constants import (X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX,
                       N_F, N_IC, N_BC, H_LEFT, H_RIGHT, X_DAM, DEVICE)


def sample_interior(n=N_F):
    """Точки внутри области."""
    x = (X_MAX - X_MIN) * torch.rand(n, 1) + X_MIN
    y = (Y_MAX - Y_MIN) * torch.rand(n, 1) + Y_MIN
    t = (T_MAX - T_MIN) * torch.rand(n, 1) + T_MIN
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)

def sample_initial(n=N_IC):
    """Точки на t = 0 (начальные условия)."""
    x = (X_MAX - X_MIN) * torch.rand(n, 1) + X_MIN
    y = (Y_MAX - Y_MIN) * torch.rand(n, 1) + Y_MIN
    t = torch.zeros(n, 1)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)

def sample_boundary(n=N_BC):
    """Точки на границе домена."""
    n4 = n // 4
    t = (T_MAX - T_MIN) * torch.rand(n4, 1) + T_MIN

    # левая
    xl = torch.full((n4, 1), X_MIN); yl = (Y_MAX - Y_MIN) * torch.rand(n4, 1) + Y_MIN
    # правая
    xr = torch.full((n4, 1), X_MAX); yr = (Y_MAX - Y_MIN) * torch.rand(n4, 1) + Y_MIN
    # нижняя
    xb = (X_MAX - X_MIN) * torch.rand(n4, 1) + X_MIN; yb = torch.full((n4, 1), Y_MIN)
    # верхняя
    xt = (X_MAX - X_MIN) * torch.rand(n4, 1) + X_MIN; yt = torch.full((n4, 1), Y_MAX)

    x = torch.cat([xl, xr, xb, xt])
    y = torch.cat([yl, yr, yb, yt])
    t = torch.cat([t, t, t, t])
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)

def initial_condition(x, y):
    """Начальные h, u, v. Здесь — dam break (скачок в x = 0)."""
    h0 = torch.where(x <= X_DAM, H_LEFT, H_RIGHT).float() # Разная высота слева и справа от точки разрыва
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(x)
    return h0, u0, v0