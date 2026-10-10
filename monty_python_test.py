# monty_python_test.py
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import pandas as pd
import seaborn as sns
import numpy as np
import math
import logging, os, sys
import time
from typing import Callable
from mp_box_mueller import Box_Muller_normal
from ahrens_dieter import (
    ahrens_dieter_cauchy,
    ahrens_dieter_exponential,
    ahrens_dieter_normal,
    generate_ahrens_dieter_samples,
)
from mp_functions import (
    Normalize_pdf,
    Compute_Mean_Std,
    Kolmogorov_Smirnov_test,
    plot_hist,
    uniform_random,
    f,
    f_est,
    g,
)

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


def test_monty_python_normal(n_samples, is_plotted):
    rng = np.random.default_rng(seed=19680801)
    normal_samples = rng.normal(0.0, 1.0, n_samples)

    montyPython_samples = [monty_python_normal() for _ in range(n_samples)]

    # Plot del histograma de los valores generados
    if is_plotted:
        plot_hist(
            montyPython_samples,
            plot=is_plotted,
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


n_samples = 1000000
rng = np.random.default_rng(seed=19680801)
normal_samples = rng.normal(0.0, 1.0, n_samples)

# Plot de la distribución normal para comparar
plot_hist(
    normal_samples,
    plot=False,
    title="Generación de valores con numpy.rng.normal",
    label="numpy.rng.normal",
    Save_plot=False,
)

# GRAND_samples = test_GRAND_normal(n_samples, False)
# Box_Muller_samples = Box_Muller_normal(n_samples, False)
# ahrens_dieter_samples = generate_ahrens_dieter_samples(n_samples)
for i in range(0, 1):
    montyPython_samples = test_monty_python_normal(n_samples, True)
    results = pd.DataFrame(
        columns=[
            # "Method",
            "mean",
            "std",
            # "ShapiroTest()",
            # "NormalTest()",
            "KS-test",
        ]
    )
    # results["Method"] = ["Teórica", "Monty Python", "Ahrens-Dieter", "Box-Muller", "GRAND"]
    results["mean"] = [
        # Compute_Mean_Std(normal_samples)[0],
        Compute_Mean_Std(montyPython_samples)[0],
        # Compute_Mean_Std(ahrens_dieter_samples)[0],
        # Compute_Mean_Std(Box_Muller_samples)[0],
        # Compute_Mean_Std(GRAND_samples)[0],
    ]
    results["std"] = [
        # Compute_Mean_Std(normal_samples)[1],
        Compute_Mean_Std(montyPython_samples)[1],
        # Compute_Mean_Std(ahrens_dieter_samples)[1],
        # Compute_Mean_Std(Box_Muller_samples)[1],
        # Compute_Mean_Std(GRAND_samples)[1],
    ]
    # results["ShapiroTest()"] = [
    # stats.shapiro(normal_samples)[1],
    # stats.shapiro(montyPython_samples)[1],
    # stats.shapiro(ahrens_dieter_samples)[1],
    # stats.shapiro(Box_Muller_samples)[1],
    # stats.shapiro(GRAND_samples)[1],
    # ]
    # results["NormalTest()"] = [
    # stats.normaltest(normal_samples)[1],
    # stats.normaltest(montyPython_samples)[1],
    # stats.normaltest(ahrens_dieter_samples)[1],
    # stats.normaltest(Box_Muller_samples)[1],
    # stats.normaltest(GRAND_samples)[1],
    # ]
    results["KS-test"] = [
        # stats.kstest((normal_samples), (normal_samples)).pvalue,
        stats.kstest((montyPython_samples), (normal_samples)).pvalue,
        # stats.kstest((ahrens_dieter_samples), (normal_samples)).pvalue,
        # stats.kstest((Box_Muller_samples), (normal_samples)).pvalue,
        # stats.kstest((GRAND_samples), (normal_samples)).pvalue,
    ]
    # print statistics as dataframe
    logging.info("n = {}. Results:\n{}".format(n_samples, results.T))
#


def benchmark_normal_generator(
    func: Callable, n_iterations: int = 1000, n_runs: int = 3
) -> tuple[float, float]:
    """
    Benchmark de una función generadora de distribución normal.

     Args:
         func: La función generadora a evaluar
         n_iteraciones: Número de valores a generar por ejecución
         n_ejecuciones: Número de veces que se ejecuta la prueba

     Devuelve:
         tupla: (tiempo_medio, tiempo_est) en segundos
    """
    times = []

    for _ in range(n_runs):
        start_time = time.perf_counter()

        if func.__name__ == "Box_Muller_normal":
            _ = func(n_iterations)
        else:
            for _ in range(n_iterations):
                _ = func()

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times), times


def plot_benchmark_results(results: dict):
    methods = list(results.keys())
    times = [results[m][2] for m in methods]
    means = [results[m][0] for m in methods]
    stds = [results[m][1] for m in methods]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, means, yerr=stds, capsize=5, color="crimson")
    plt.xticks(rotation=45)
    plt.ylabel("Tiempo (segundos)")
    plt.title("Comparación de métodos para generar distribuciones normales")
    plt.tight_layout()
    # ax = plt.gca()
    # ax.set_xlim([-3, 3])
    # ax.set_ylim([0, 2.85])
    # plt.savefig(current + "/images/Benchmark_Test.png")
    plt.show()


# Config benchmark
n_iterations = 1000
n_runs = 10

results = {
    "Monty Python": benchmark_normal_generator(
        monty_python_normal, n_iterations, n_runs
    ),
    # "GRAND": benchmark_normal_generator(GRAND_normal, n_iterations, n_runs),
    # "Box-Muller": benchmark_normal_generator(Box_Muller_normal, n_iterations, n_runs),
    # "Ahrens-Dieter": benchmark_normal_generator(
    # lambda: ahrens_dieter_normal()[0], n_iterations, n_runs
    # ),
}

# Print results
print(f"\nBenchmark resultados ({n_iterations:,} iteraciones, {n_runs} ejecuciones):\n")
for method, (mean_time, std_time, times) in results.items():
    print(f"{method:15} {mean_time:.7f} ± {std_time:.7f} segundos")
    logging.warning(f"{method:15} {mean_time:.7f} ± {std_time:.7f} segundos")

# plot_benchmark_results(results)
