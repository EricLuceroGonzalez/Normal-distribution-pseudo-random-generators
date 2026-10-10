import numpy as np
import statistics as stats
import math
import matplotlib.pyplot as plt
import logging, os, sys

sys.path.append(os.path.abspath("../"))
current = os.path.dirname(os.path.realpath(__file__))


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
