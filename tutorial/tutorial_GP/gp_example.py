import numpy as np
from utils_plot import plot_gpr_samples

"""
example to see how a GP works. 
start from a sin(x)/x signal, see how the GP reconstructs it from samples. 
"""

# get real sin(x)/x data
X = np.linspace(start=1, stop=15, num=1000).reshape(-1, 1)
y = np.squeeze(np.sin(X) / X)
x_plot1 = np.linspace(start=1, stop=15, num=1000).reshape(-1, 1)
y_plot1 = np.squeeze(np.sin(x_plot1) / x_plot1)
import matplotlib.pyplot as plt

plt.plot(x_plot1, y_plot1, label=r"$f(x) = \sin(x) / x$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")

# find a subset of samples to train the GP
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=5, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# GP fitting
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

# plots
fig, ax = plt.subplots()
plot_gpr_samples(gaussian_process, n_samples=7, ax=ax)
plt.scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
plt.grid()
plt.legend()

# predict the fitted function from the GP
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.figure()
plt.plot(X, y, label=r"$f(x) = \sin(x) / x$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")

#####################
# NOISY DATASET
#####################
# do the same w/ a noisy dataset
noise_std = 0.1
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

plt.figure()
plt.plot(x_plot1, y_plot1, label=r"$f(x) = sin(x)/x$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.grid()
plt.legend()

# GP fitting
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)

# new data
x_plot = np.linspace(start=0, stop=15, num=1_000).reshape(-1, 1)
# GP prediction
mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

plt.figure()
plt.plot(x_plot1, y_plot1, label=r"$f(x) = sin(x)/x$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(x_plot, mean_prediction, label="Mean prediction")
plt.fill_between(
    x_plot.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
# _ = plt.title("Gaussian process regression on a noisy dataset")
plt.grid()
plt.show()

