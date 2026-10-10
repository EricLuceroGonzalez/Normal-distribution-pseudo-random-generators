# mp_qq_plot.py
import numbers
from mp_functions import uniform_random, g, f
import numpy as np
import matplotlib.pyplot as plt


def monty_python_normal():
    b = np.sqrt(2 * np.pi)
    a = np.sqrt(np.log(4))
    print(a, b)

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


def qqplot(
    x,
    y,
    quantiles=None,
    interpolation="nearest",
    ax=None,
    rug=False,
    rug_length=0.05,
    rug_kwargs=None,
    **kwargs,
):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))
        print("----1----")
        print(quantiles)

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        print("----2----")
        print(quantiles)
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
        print("----3----")
        print(quantiles)
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)
    print(f" len = {len(x_quantiles)}")
    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c="gray", alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c="gray", alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    coef = np.polyfit(x_quantiles, y_quantiles, 1)
    poly1d_fn = np.poly1d(coef)
    print(coef)
    ax.scatter(x_quantiles, y_quantiles, **kwargs, cmap="viridis")
    ax.plot(x_quantiles, y_quantiles, x_quantiles, poly1d_fn(x_quantiles), "--k")
    #'--k'=black dashed line, 'yo' = yellow circle marker

    # Setup


n_samples = 100
rng = np.random.default_rng(seed=19680801)
normal_samples = rng.normal(0.0, 1.0, n_samples)
exp_samples = rng.exponential(3.83, n_samples)

# Example 1: samples of the same length
n = 100  # Number of samples to draw
x = normal_samples  # Sample 1: X ~ N(0, 1)
mp_samples_raw = [monty_python_normal() for _ in range(n_samples)]
mp_samples = np.array(mp_samples_raw)
# y = exp_samples
y = mp_samples
# rng.standard_t(df=5, size=n)  # Sample 2: Y ~ t(5)


# poly1d_fn is now a function which takes in x and returns an estimate for y


# Draw quantile-quantile plot

plt.figure()
qqplot(x, y, alpha=0.5, edgecolor="k")
plt.xlabel("X")
plt.ylabel("Y")
# plt.title("QQ-plot")
plt.grid(linestyle="--", alpha=0.5)
plt.show()
plt.close()

# # Example 2: samples of different lengths
# n_u = 50  # Number of U samples to draw
# n_v = 100  # Number of V samples to draw
# u = rng.normal(size=n_u)  # Sample 1: U ~ N(0, 1)
# v = rng.standard_t(df=5, size=n_v)  # Sample 2: V ~ t(5)

# # Draw quantile-quantile plot
# plt.figure()
# qqplot(u, v, c="r", alpha=0.5, edgecolor="k")
# plt.xlabel("U")
# plt.ylabel("V")
# plt.title("Two samples of different lengths")
# plt.show()
# plt.close()

# # Draw quantile-quantile plot with rug plot
# plt.figure()
# qqplot(u, v, c="r", alpha=0.5, edgecolor="k", rug=True)
# plt.xlabel("U")
# plt.ylabel("V")
# plt.title("Two samples of different lengths, with rug plot")
# plt.show()
# plt.close()
