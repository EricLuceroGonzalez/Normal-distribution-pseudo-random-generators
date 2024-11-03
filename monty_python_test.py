# monty_python_test.py
from fitter import Fitter, get_common_distributions
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import pandas as pd

import numpy as np
import math
import logging, os, sys

sys.path.append(os.path.abspath("../"))
current = os.path.dirname(os.path.realpath(__file__))
clear = lambda: os.system("clear")
clear()

# Imprime logs en el archivo MONTY_PYTHON.log
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=current + "/MONTY_PYTHON.log",
    encoding="utf-8",
    filemode="a",
    format="%(levelname)s:%(message)s",
    level=logging.DEBUG,
    # level=logging.WARNING,
)
logging.getLogger("matplotlib.font_manager").disabled = True
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.debug(today)


def Normalize_pdf(data):
    sum_each = np.sum(data)
    normalize = [i / sum_each for i in data]
    return normalize


def Compute_Mean_Std(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    return data_mean, data_std


def Kolmogorov_Smirnov_test(data_1, data_2, name=""):
    ks_test = stats.kstest(data_1, data_2)
    # Return p-values
    return ks_test.pvalue


def plot_hist(
    data, plot=False, title="", label="", Save_plot=False, plot_color="crimson"
):
    plt.figure(figsize=(10, 6))
    # Plot de la distribución normal para comparar (en rojo)
    x = np.linspace(-4, 4, len(data))
    # Numero de columnas en el histograma:
    # bins_number = math.ceil(math.sqrt(len(data)))
    bins_number = math.ceil(1 + 3.322 * math.log(len(data)))
    plt.hist(
        data,
        bins=bins_number,
        density=True,
        alpha=0.75,
        color=plot_color,
        edgecolor="white",
        label="{}".format(label),
    )
    plt.plot(x, f_est(x), label=r"$N(0,1) = \frac{1}{\sqrt{2*\pi}} * e^{(-0.5x^2)}$")

    ax = plt.gca()
    # ax.set_xlim([-3, 3])
    # ax.set_ylim([0, 0.5])
    plt.legend()
    plt.title("{}, n = {:,}".format(title, len(data), ","))

    plt.xlabel("$x$")
    plt.ylabel("Frecuencia")
    plt.grid(False)
    if Save_plot:
        plt.savefig(current + "/images/{}_{}_samples.png".format(label, len(data)))
    if plot:
        plt.show() if plot else ""
        plt.close()


def uniform_random():
    return np.random.uniform(0, 1)


# Función de densidad absoluta
def f(x):
    return 2 * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# The rotated and stretched density for the tail region (Monty-Python)
def g(x, s, b):
    return 1 / b - s * (f(s * (b - x)) - 1 / b)


# Densidad estándar
def f_est(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


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
            return b * (B / c + C)
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


def monty_python_normal():
    b = np.sqrt(2 * np.pi)
    a = np.sqrt(np.log(4))

    while True:
        x = uniform_random() * b
        # Region F (central part)
        if x < a:
            return np.random.choice([-1, 1]) * x

        y = uniform_random() / b
        # Region G (normal distribution curve)
        if y < f(x):
            return np.random.choice([-1, 1]) * x

        s = a / (b - a)

        # Region H (rotated and stretched)
        if y > g(x, s, b):
            return np.random.choice([-1, 1]) * s * (b - x)

        # Tail case
        while True:
            U1 = uniform_random()
            U2 = uniform_random()
            x_tail = -np.log(U1) / b
            y_tail = -np.log(U2)
            if y_tail + y_tail > x_tail * x_tail:
                return np.random.choice([-1, 1]) * (b + x_tail)


def GRAND_normal():
    i = 0
    x = uniform_random()
    while x < 0.5:
        x *= 2
        i += 1
    while True:
        u = (
            norm.ppf(1 - 2 ** (-(i + 1) - 1)) - norm.ppf(1 - 2 ** (-i - 1))
        ) * uniform_random()
        v = u * (u / 2 + norm.ppf(1 - 2 ** (-i - 1)))
        while True:
            if v < uniform_random():
                if uniform_random() < 0.5:
                    return norm.ppf(1 - 2 ** (-i - 1)) + u
                else:
                    return -1 * norm.ppf(1 - 2 ** (-i - 1)) - u
            else:
                v = uniform_random()
            if v < uniform_random():
                break


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


def test_monty_python_normal(n_samples, is_plotted):
    rng = np.random.default_rng(seed=19680801)
    normal_samples = rng.normal(0.0, 1.0, n_samples)

    montyPython_samples = [monty_python_normal() for _ in range(n_samples)]

    # Plot del histograma de los valores generados
    if is_plotted:
        plot_hist(
            montyPython_samples,
            plot=True,
            title="Generación de valores con Monty Python",
            label="Monty Python",
            Save_plot=False,
            plot_color="dodgerblue",
        )

    return np.array(montyPython_samples)


def test_GRAND_normal(n_samples, is_plotted):
    grand_samples = [GRAND_normal() for _ in range(n_samples)]
    if is_plotted:
        plot_hist(
            grand_samples,
            True,
            "GRAND Distribución Normal",
            "GRAND_normal",
            Save_plot=False,
        )
    return grand_samples


n_samples = 100000
rng = np.random.default_rng(seed=19680801)
normal_samples = rng.normal(0.0, 1.0, n_samples)

# Plot de la distribución normal para comparar
plot_hist(
    normal_samples,
    plot=True,
    title="Generación de valores con numpy.rng.normal",
    label="numpy.rng.normal",
    Save_plot=False,
)

montyPython_samples = test_monty_python_normal(n_samples, True)
Box_Muller_samples = Box_Muller_normal(n_samples, True)
GRAND_samples = test_GRAND_normal(n_samples, True)
ahrens_dieter_samples = generate_ahrens_dieter_samples(n_samples)
results = pd.DataFrame(
    columns=[
        "Method",
        "mean",
        "std",
        "ShapiroTest()",
        "NormalTest()",
        "KS-test",
    ]
)
results["Method"] = ["Teórica", "Monty Python", "Ahrens-Dieter", "Box-Muller", "GRAND"]
results["mean"] = [
    Compute_Mean_Std(normal_samples)[0],
    Compute_Mean_Std(montyPython_samples)[0],
    Compute_Mean_Std(ahrens_dieter_samples)[0],
    Compute_Mean_Std(Box_Muller_samples)[0],
    Compute_Mean_Std(GRAND_samples)[0],
]
results["std"] = [
    Compute_Mean_Std(normal_samples)[1],
    Compute_Mean_Std(montyPython_samples)[1],
    Compute_Mean_Std(ahrens_dieter_samples)[1],
    Compute_Mean_Std(Box_Muller_samples)[1],
    Compute_Mean_Std(GRAND_samples)[1],
]
results["ShapiroTest()"] = [
    stats.shapiro(normal_samples)[1],
    stats.shapiro(montyPython_samples)[1],
    stats.shapiro(ahrens_dieter_samples)[1],
    stats.shapiro(Box_Muller_samples)[1],
    stats.shapiro(GRAND_samples)[1],
]
results["NormalTest()"] = [
    stats.normaltest(normal_samples)[1],
    stats.normaltest(montyPython_samples)[1],
    stats.normaltest(ahrens_dieter_samples)[1],
    stats.normaltest(Box_Muller_samples)[1],
    stats.normaltest(GRAND_samples)[1],
]
results["KS-test"] = [
    stats.kstest((normal_samples), (normal_samples)).pvalue,
    stats.kstest((montyPython_samples), (normal_samples)).pvalue,
    stats.kstest((ahrens_dieter_samples), (normal_samples)).pvalue,
    stats.kstest((Box_Muller_samples), (normal_samples)).pvalue,
    stats.kstest((GRAND_samples), (normal_samples)).pvalue,
]
# print statistics as dataframe
logging.info("n = {}. Results:\n{}".format(n_samples, results.T))
