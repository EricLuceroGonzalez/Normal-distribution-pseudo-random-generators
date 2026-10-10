# mp_KS_test.py
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
from mp_functions import uniform_random, g, f, plot_hist
import logging, os, sys

sys.path.append(os.path.abspath("../"))
current = os.path.dirname(os.path.realpath(__file__))


def Box_Muller_normal(n_samples=1, is_plotted=False):
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
    return z1[0]


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


def cdf(sample, x, sort=False):
    # Sorts the sample, if unsorted
    if sort:
        sample.sort()
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    return cdf


def ks_norm(sample):
    # Sorts the sample
    sample.sort()
    # Evaluates the KS statistic
    D_ks = []  # KS Statistic list
    for x in sample:
        cdf_normal = stats.norm.cdf(x=x, loc=0, scale=1)
        cdf_sample = cdf(sample=sample, x=x)
        D_ks.append(abs(cdf_normal - cdf_sample))
    ks_stat = max(D_ks)
    # Calculates the P-Value based on the two-sided test
    # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
    p_value = stats.kstwo.sf(ks_stat, len(sample))
    return {"ks_stat": ks_stat, "p_value": p_value}


# Calcula el estadístico de Kolmogorov-Smirnov (KS).
def ks_norm(sample):
    # Ordena la muestra de menor a mayor.
    sample.sort()
    D_ks = []
    for x in sample:
        # Evalúa la CDF de la distribución normal estándar.
        cdf_normal = stats.norm.cdf(x=x, loc=0, scale=1)
        # Evalúa la CDF empírica basada en la muestra.
        cdf_sample = cdf(sample=sample, x=x)
        # Diferencia entre la CDF teórica y la empírica.
        D_ks.append(abs(cdf_normal - cdf_sample))
    # Estadístico KS.
    ks_stat = max(D_ks)

    # Calcula el valor p (P-Value) basado en la prueba.
    p_value = stats.kstwo.sf(ks_stat, len(sample))
    return {"ks_stat": ks_stat, "p_value": p_value}


def Standarization(arg):
    mean = np.mean(arg)
    stdv = np.std(arg)
    normalized = []
    for i in arg:
        normalized.append((i - mean) / (stdv))
    return mean, stdv, normalized


def standardize(sample):

    # Standardize the data
    mean = sample.mean()
    std = sample.std()
    sample_z = (sample - mean) / std

    return sample_z


# Create random samples
n_samples = 100
norm_a = np.random.normal(loc=0, scale=1, size=n_samples)
norm_b = np.random.normal(loc=0.1, scale=1, size=n_samples)
norm_c = np.random.normal(loc=3, scale=1, size=n_samples)
rng = np.random.default_rng(seed=19680801)
normal_samples = rng.normal(0.0, 1.0, 100)
mp_samples_raw = [monty_python_normal() for _ in range(n_samples)]
bm_samples_raw = [Box_Muller_normal() for _ in range(n_samples)]
mp_samples = np.array(mp_samples_raw)
bm_samples = np.array(bm_samples_raw)

norm_a.sort()
norm_b.sort()
norm_c.sort()
normal_samples.sort()
mp_samples.sort()
bm_samples.sort()

# Creates their CDFs for visualization
cdf_norm_a = np.array([cdf(norm_a, x, sort=True) for x in norm_a])
cdf_norm_b = np.array([cdf(norm_b, x, sort=True) for x in norm_b])
cdf_norm_c = np.array([cdf(norm_c, x, sort=True) for x in norm_c])
cdf_normal_samples = np.array(
    [cdf(normal_samples, x, sort=True) for x in normal_samples]
)
cdf_mp = np.array([cdf(mp_samples, x, sort=True) for x in mp_samples])
cdf_bm = np.array([cdf(bm_samples, x, sort=True) for x in bm_samples])
# Calculates the KS statistic
# We wish to test for normality, so we want to compare to a normal distribution with mean = 0 and variance = 1.
# We need then to standardize the data. If we don't do that, a normal distribution with other mean or variance will fail the test.
# Performs the KS normality test in the samples
ks_norm_a = ks_norm(standardize(norm_a))
ks_norm_b = ks_norm(standardize(norm_b))
ks_norm_c = ks_norm(standardize(norm_c))
ks_normal_samples = ks_norm(standardize(normal_samples))
ks_mp = ks_norm(standardize(mp_samples))
ks_bm = ks_norm(standardize(bm_samples))
# Prints the result
print(
    f"norm_a: ks = {ks_norm_a['ks_stat']:.4f} (p-value = {ks_norm_a['p_value']:.3e}, is normal = {ks_norm_a['p_value'] > 0.05})"
)
print(
    f"norm_b: ks = {ks_norm_b['ks_stat']:.4f} (p-value = {ks_norm_b['p_value']:.3e}, is normal = {ks_norm_b['p_value'] > 0.05})"
)
print(
    f"norm_c: ks = {ks_norm_c['ks_stat']:.4f} (p-value = {ks_norm_c['p_value']:.3e}, is normal = {ks_norm_c['p_value'] > 0.05})"
)
print(
    f"normal_samp: ks = {ks_normal_samples['ks_stat']:.4f} (p-value = {ks_normal_samples['p_value']:.3e}, is normal = {ks_normal_samples['p_value'] > 0.05})"
)
print(
    f"Monty Python: ks = {ks_mp['ks_stat']:.4f} (p-value = {ks_mp['p_value']:.3e}, is normal = {ks_mp['p_value'] > 0.05})"
)
print(
    f"Box Muller: ks = {ks_bm['ks_stat']:.4f} (p-value = {ks_bm['p_value']:.3e}, is normal = {ks_bm['p_value'] > 0.05})"
)

plt.figure(figsize=(10, 6))
# Plot de la distribución normal para comparar (en rojo)
x = np.linspace(-5, 5, len(norm_a))
# Numero de columnas en el histograma:
# bins_number = math.ceil(math.sqrt(len(data)))
bins_number = math.ceil(1 + 3.322 * math.log(len(norm_a)))
plt.hist(
    norm_a,
    bins=bins_number,
    density=True,
    alpha=0.75,
    edgecolor="white",
    label="{}".format("norm_a"),
)
plt.hist(
    norm_b,
    bins=bins_number,
    density=True,
    alpha=0.75,
    edgecolor="white",
    label="{}".format("norm_b"),
)
plt.hist(
    norm_c,
    bins=bins_number,
    density=True,
    alpha=0.75,
    edgecolor="white",
    label="{}".format("norm_c"),
)
plt.hist(
    normal_samples,
    bins=bins_number,
    density=True,
    alpha=0.75,
    edgecolor="white",
    label="{}".format("normal_samp"),
)
plt.hist(
    mp_samples,
    bins=bins_number,
    density=True,
    alpha=0.75,
    edgecolor="white",
    label="{}".format("Monty Python"),
)


plt.legend()
# plt.show()


# Visualize them
# plt.figure(figsize=(10, 8))
# # sns.lineplot(x=norm_a, y=cdf_norm_a, color="b", label="norm_a")
# # sns.lineplot(x=norm_b, y=cdf_norm_b, color="g", label="norm_b")
# # sns.lineplot(x=norm_c, y=cdf_norm_c, color="r", label="norm_c")
# sns.lineplot(x=mp_samples, y=cdf_mp, color="crimson", label="Monty Python")
# sns.lineplot(
#     x=normal_samples,
#     y=cdf_normal_samples,
#     color="dodgerblue",
#     label="rng.normal(0.0, 1.0, n_samples) - (seed=19680801)",
# )
# # plt.legend(["norm_a", "norm_b", "norm_c"])
# plt.legend()
# # plt.show()


def plot_CDF(dist1, dist2, cdf_1, cdf_2):
    plt.figure(figsize=(10, 8))
    sns.lineplot(
        x=dist2,
        y=cdf_2,
        color="black",
        label="rng.normal(0.0, 1.0, n_samples) - (seed=19680801)",
    )
    sns.lineplot(x=dist1, y=cdf_1, color="crimson", label="Monty Python")
    min_x = int(min(min(dist1), min(dist2)))
    max_x = int(max(max(dist1), max(dist2)))
    print(min_x, max_x)
    for i in range(min_x * 3, max_x * 3):
        j = i / 3
        plt.plot(
            [j, j], [cdf(dist1, j), cdf(dist2, j)], color="dodgerblue", linewidth=2.5
        )
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.title(
        f"Prueba de Kolmogorov-Smirnov - p-value: {round(ks_mp['p_value'],3)}, n = {n_samples}"
    )
    # plt.savefig(
    #     current
    #     + f"/Images/Kolmogorov_Smirnov-test-MP-p-value:{round(ks_mp['p_value'],3)}.png"
    # )
    # plt.show()


# plot_CDF(mp_samples, normal_samples, cdf_1=cdf_mp, cdf_2=cdf_normal_samples)
# plot_CDF(bm_samples, normal_samples, cdf_1=cdf_bm, cdf_2=cdf_normal_samples)
# plot_hist(
#     mp_samples,
#     True,
#     Save_plot=True,
#     title="Generación de valores con Monty Python",
#     label="Monty Python",
# )
# Ordena los datos para comparar con cuantiles teóricos
rng = np.random.default_rng(seed=19680801)
normal_samples = rng.normal(0.0, 1.0, 1000)
mp_samples_raw = [monty_python_normal() for _ in range(n_samples)]
mp_samples = np.array(mp_samples_raw)
sorted_data = np.sort(normal_samples)
quantiles_theoretical = stats.norm.ppf(
    np.linspace(
        0.5 / len(normal_samples), 1 - 0.5 / len(normal_samples), len(normal_samples)
    )
)

# Calcula la línea de referencia (cuantiles teóricos contra sí mismos)
slope, intercept, r_value = stats.linregress(quantiles_theoretical, sorted_data)
line = slope * quantiles_theoretical + intercept

# Genera el gráfico QQ-plot
plt.figure(figsize=(8, 6))
plt.scatter(quantiles_theoretical, sorted_data, alpha=0.6, label="Datos simulados")
plt.plot(quantiles_theoretical, line, color="red", label="Línea de referencia")
plt.axline((0, 0), slope=1, color="green", linestyle="--", label="Línea ideal (x = y)")
plt.title("QQ-Plot: Comparación con distribución normal estándar")
plt.xlabel("Cuantiles teóricos")
plt.ylabel("Cuantiles de los datos")
plt.legend()
plt.grid(alpha=0.4)
plt.show()
# efault_rng(seed=19680801)
# normal_samples = rng.normal(0.0, 1.0, n_samples) - (seed=19680801)
