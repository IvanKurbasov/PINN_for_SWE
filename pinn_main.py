from pinn_model import PINN
from swe_plotting import plot_history, plot_hx_with_constant_y, plot_hll_with_constant_y


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # model = PINN()
    # model, history = model.pinn_train() # Обучение модели
    #
    # plot_history(history) # Построение всех Loss модели
    # plot_hx_with_constant_y(model) # Построение уровня воды в разные моменты времени
    plot_hll_with_constant_y() # Построение уровня воды в один момент времени для численного решения





