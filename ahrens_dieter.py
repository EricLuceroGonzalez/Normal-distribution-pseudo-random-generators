# ahrens_dieter.py
import numpy as np
from mp_functions import uniform_random, plot_hist


def ahrens_dieter_exponential():
    """
    From: Gaussian random number generators (2007). David B. Thomas et al.
    Ahrens-Dieter method (Algoritmo 7)
    """
    # Constantes
    A = 0.0
    B = 1.0
    C = 1.0
    P = 0.98
    LN2 = np.log(2)

    while True:
        x = uniform_random()
        a = A

        while x < 0.5:
            x = 2 * x
            a = a + LN2

        x = x - 1

        if x < P:  # Primera rama (ejecuta ~98% de las veces)
            return a + B / (C - x)
        else:
            while True:
                u1 = uniform_random()
                u2 = uniform_random()
                if u2 <= np.exp(-u1):
                    return a + u1


def ahrens_dieter_cauchy():
    """
    Valores Cauchy Ahrens-Dieter method (Algoritmo 8)
    """
    A = 1.0
    B = 1.0
    C = 0.0

    while True:
        u1 = uniform_random()
        b = u1 - 0.5
        c = A - (u1 - 0.5) ** 2

        if c > 0:  # Primera rama (ejecuta ~99.8% de las veces)
            return b * ((B / c) + C)
        else:
            while True:
                u1 = uniform_random()
                u2 = uniform_random()
                v = np.pi * (u1 - 0.5)
                y = np.tan(v)
                if u2 <= (1 + y * y) * np.exp(-0.5 * y * y):
                    return y


def ahrens_dieter_normal():
    """
    Generar un par de números aleatorios normales estándar independientes
    utilizando el método sin tablas de Ahrens-Dieter con generadores
    exponenciales y de Cauchy. (Algoritmo 9).
    """

    s = 1 if uniform_random() < 0.5 else -1

    x = ahrens_dieter_exponential()
    y = ahrens_dieter_cauchy()

    z = np.sqrt((2 * x) / (1 + y * y))

    return (s * z, y * z)


def generate_ahrens_dieter_samples(n_samples):
    """
    Genera n muestras de la distribución normal.
    Nota: Cada llamada a ahrens_dieter_normal() genera 2 números,
    así que sólo necesitamos n/2 llamadas.
    """
    samples = []
    for _ in range(n_samples // 2 + n_samples % 2):
        x, y = ahrens_dieter_normal()
        samples.extend([x, y])
    plot_hist(
        samples,
        False,
        "Ahrens-Dieter Distribución Normal",
        "ahrens_dieter_normal",
        Save_plot=False,
    )

    return samples[:n_samples]
