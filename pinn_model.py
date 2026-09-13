import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from swe_sampler import sample_interior, sample_initial, sample_boundary, initial_condition
from SWE import residual
import swe_constants

from swe_constants import *

class Sin(nn.Module):
    """Implements `sin` activation function"""

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
    def __init__(self, hidden=128, depth=10):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for i in range(depth - 1):
            if i % 2 == 0:
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            else:
                layers += [nn.Linear(hidden, hidden), Sin()]
        layers += [nn.Linear(hidden, 3)] # На выходе h, u, v
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            Sin(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            Sin(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x, y, t):
        # x,y,t - векторы (N, 1), получу (N, 3)
        input = torch.cat([x, y, t], dim=1)
        out = self.net(input) # (N, 3)
        h = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v

    def pinn_train(self):
        optim = torch.optim.Adam(self.parameters(), lr=swe_constants.LR)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=DECAY_EVERY, gamma=LR_DECAY)

        history = {"loss": [], "pde": [], "ic": [], "bc": [], "pos": []}

        pbar = tqdm(range(EPOCHS), desc="Training")
        for epoch in pbar:
            self.train()
            optim.zero_grad()

            x_f, y_f, t_f = sample_interior()
            R = residual(self, x_f, y_f, t_f)
            L_pde = torch.mean(R ** 2)  # Невязка внутри области

            x_i, y_i, t_i = sample_initial()
            h_pred, u_pred, v_pred = self(x_i, y_i, t_i)
            h_true, u_true, v_true = initial_condition(x_i, y_i)
            L_ic = (torch.mean((h_pred - h_true) ** 2)
                    + torch.mean((u_pred - u_true) ** 2)
                    + torch.mean((v_pred - v_true) ** 2))

            x_b, y_b, t_b = sample_boundary()
            h_b, u_b, v_b = self(x_b, y_b, t_b)
            # На границе u = v = 0 (Dirichlet)
            L_bc = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)

            # !!! ВОТ ТУТ МОЖНО ЧУТЬ ОПТИМИЗИРОВАТЬ
            current_h, _, _ = self(x_f, y_f, t_f)
            L_pos = torch.mean(torch.relu(-current_h) ** 2)

            loss = (W_PDE * L_pde
                    + W_IC * L_ic
                    + W_BC * L_bc
                    + W_POS * L_pos)

            loss.backward()
            optim.step()
            sched.step()

            history["loss"].append(loss.item())
            history["pde"].append(L_pde.item())
            history["ic"].append(L_ic.item())
            history["bc"].append(L_bc.item())
            history["pos"].append(L_pos.item())

            if epoch % 100 == 0:
                pbar.set_description(
                    f"L={loss.item():.3e} "
                    f"PDE={L_pde.item():.3e} "
                    f"IC={L_ic.item():.3e} "
                    f"BC={L_bc.item():.3e}"
                )

        torch.save(self.state_dict(), "data/pinn_swe.model")
        return self, history
