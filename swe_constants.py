import torch

g = 9.81

X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -10.0, 10.0
T_MIN, T_MAX = 0.0, 1.0

EPOCHS = 200
BATCH_SIZE = 2048
LR = 1e-3
LR_DECAY = 0.1
DECAY_EVERY = 2000

# Число точек каждого типа
N_F = 8192     # внутри области (PDE)
N_IC = 2048    # начальные условия
N_BC = 2048    # граничные условия

# Веса loss-компонент
W_PDE = 1.0
W_IC = 100.0    # начальные условия
W_BC = 1.0      # конечные условия
W_POS = 10.0    # неотрицательность глубины

H_LEFT = 2.0
H_RIGHT = 1.0
X_DAM = 0.0

CFL = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")