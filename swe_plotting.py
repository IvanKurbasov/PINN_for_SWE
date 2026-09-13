import torch

import matplotlib.pyplot as plt
from swe_constants import T_MIN, T_MAX, X_MIN, X_MAX, DEVICE
from pinn_model import PINN

from hll_solver_for_swe import solve_2d_dam_break

def plot_history(history, save_path="data/loss.png"):
    plt.figure(figsize=(10, 6))

    plt.semilogy(history["loss"], label="Итоговый Loss", linewidth=2)
    plt.semilogy(history["pde"],  label="Невязка точек внутри области",  alpha=0.7)
    plt.semilogy(history["ic"],   label="Невязка начальных условий",   alpha=0.7)
    plt.semilogy(history["bc"],   label="Невязка граничных условий",   alpha=0.7)
    plt.semilogy(history["pos"],  label="Невязка отрицательной высоты воды",  alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN training history")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_hx_with_constant_y(model, times=torch.linspace(T_MIN,T_MAX,10), save_path="data/h_profile.png"):
    model = PINN().to(DEVICE)  # ← та же архитектура, что при обучении
    model.load_state_dict(torch.load("data/pinn_swe.model", map_location=DEVICE))
    model.eval()

    x = torch.linspace(X_MIN, X_MAX, 400).view(-1, 1).to(DEVICE)
    y = torch.zeros_like(x).to(DEVICE)

    plt.figure(figsize=(10, 6))

    with torch.no_grad():
        for t_val in times:
            t = torch.full_like(x, t_val).to(DEVICE)
            h, u, v = model(x, y, t)
            h_np = h.cpu().numpy().flatten()
            x_np = x.cpu().numpy().flatten()
            plt.plot(x_np, h_np, label=f"t = {t_val:.1f}")

    plt.xlabel("x")
    plt.ylabel("h (глубина)")
    plt.title("Профиль h(x, y=0, t) в разные моменты")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_hll_with_constant_y(Nt=20):
    history, X, times = solve_2d_dam_break()

    H_Nt = history[Nt]
    t = times[Nt]
    j_mid = H_Nt.shape[1] // 2

    plt.figure(figsize=(10, 6))
    plt.plot(X[:, j_mid], H_Nt[:, j_mid], 'b-', linewidth=2, label="HLL, y=0")
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title("Профиль h(x, y=0) в t = " + str(round(t, 2)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/hll_profile_30.png", dpi=150)
    plt.show()