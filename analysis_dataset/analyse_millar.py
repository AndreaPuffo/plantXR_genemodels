"""
data from a range of different publications (see Millar email)

Quantitative analysis of regulatory flexibility under changing environmental conditions
Kieron D Edwards, Ozgur E Akman, Kirsten Knox, Peter J Lumsden, Adrian W Thomson, Paul E Brown, Alexandra Pokhilko, Laszlo Kozma‐Bognar, Ferenc Nagy, David A Rand, and Andrew J Millar

Using higher-order dynamic bayesian networks to model periodic data from the circadian clock of arabidopsis thaliana",
Daly, Rónán, Kieron D. Edwards, John S. O’Neill, Stuart Aitken, Andrew J. Millar, and Mark Girolami,
International Conference on Pattern Recognition in Bioinformatics, 2009

Dataset found at
https://biodare2.ed.ac.uk/experiment/12730555174222/data/view/ts

this script loads the dataset,  and fits a GP to the data
considers only one gene (just to see how GP performs)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv('../datasets/12730555174222_data.LIN_DTR.format.csv', delimiter=',')
# dataset params
n_genes = 15
time = data.values[:, 0]
min_time, max_time = min(time), max(time)
y_main = data.values[:, 1:1+n_genes]
og_mean, og_std = y_main.mean(axis=0), y_main.std(axis=0)
y_main = (y_main - y_main.mean(axis=0))/y_main.std(axis=0)
# X_train = np.linspace(start=0, stop=13*4, num=13).reshape(-1, 1)
X_train = np.tile(time, n_genes).reshape((n_genes, 95)).T

plt.figure()
for idx in range(n_genes):
    # plt.figure()
    plt.plot(data.values[0:50:5, 0], y_main[0:50:5, idx])
plt.grid()
plt.xlabel('Time [hours]')
plt.ylabel('Gene CCA1 Expression (normalised)')

plt.figure()
tmp = data.values[:, 140:150]
tmp = (tmp - tmp.mean(axis=0) ) / tmp.std(axis=0)
plt.plot(data.values[0:50:5, 0], tmp[0:50:5])
plt.grid()
plt.xlabel('Time [hours]')
plt.ylabel('Gene ELF3 Expression (normalised)')


plt.figure()
tmp = data.values[:, 160:170]
tmp = (tmp - tmp.mean(axis=0) ) / tmp.std(axis=0)
plt.plot(data.values[0:50:5, 0], tmp[0:50:5])
plt.grid()
plt.xlabel('Time [hours]')
plt.ylabel('Gene TOC1 Expression (normalised)')


# compute some sort of weighted std from water data for other works
for idx in range(len(time)):
    print( np.nanstd(y_main[idx,:]) )


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# GP regression and plots
kernel = 1 * RBF(length_scale=1.0,
                 length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=0.001**2,
                                            n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_main)


x_plot = np.linspace(start = min_time, stop = max_time, num = 1000).reshape(-1, 1)
x_plot = np.tile(x_plot, (n_genes, ))
mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

plt.figure()
plt.scatter(X_train, y_main, label=r"samples")
# for i in range(obj.shape[1]):
#     plt.scatter(X_train, (obj[:, i]-np.nanmean(obj[:, i]))/og_std, label="Observations")
plt.plot(x_plot, mean_prediction, label="Mean prediction ")
for idx in range(n_genes):
    plt.fill_between(
        x_plot[:, idx].ravel(),
        mean_prediction[:, idx] - 19.6 * std_prediction[:, idx],
        mean_prediction[:, idx] + 19.6 * std_prediction[:, idx],
        alpha=0.5,
        label=r"95% confidence ",
    )
plt.legend()
plt.show()
