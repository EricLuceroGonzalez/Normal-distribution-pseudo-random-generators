# mp_box_mueller.py
import numpy as np
from mp_functions import plot_hist


def Box_Muller_normal(n_samples=10, is_plotted=False):
    # https://github.com/MrinalRajak/Box_Muller_Transform/blob/main/Box-Muller%20Transform.py

    x1 = np.random.random(n_samples)
    x2 = np.random.random(n_samples)

    r = np.sqrt(-2 * np.log(x1))
    theta = 2 * np.pi * x2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)

    # Plot del histograma de los valores generados
    if is_plotted:
        plot_hist(
            z1,
            plot=True,
            title="Generación de valores con Box-Muller",
            label=r"$r\cos(\theta)$",
            Save_plot=False,
        )
    return z1
