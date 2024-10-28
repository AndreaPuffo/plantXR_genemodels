"""# data from
Circadian expression of genes: modelling the Arabidopsis circadian clock
 	Edwards K, Millar A, Townsend H, Emmerson Z, Schildknecht B.
found at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5612
this script loads the dataset, and fits a GP to the data
considers only one gene (just to see how GP performs)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv('../datasets/gse5612_redux.txt', delimiter=';', index_col=0)
# define dataset params
n_genes = 4
y_main = data.values[0:n_genes,:].T
X_train = np.linspace(start=0, stop=13*4, num=13).reshape(-1, 1)
X_train = np.tile(X_train, (n_genes, ))
X = X_train

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# GP regression + plots
kernel = 1 * RBF(length_scale=1.0,
                 length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2,
                                            n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_main)


x_plot = np.linspace(start = 0, stop = 13*4, num = 100).reshape(-1, 1)
x_plot = np.tile(x_plot, (n_genes, ))
mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

plt.scatter(X, y_main, label=r"samples")
plt.plot(x_plot, mean_prediction, label="Mean prediction ")
for idx in range(n_genes):
    plt.fill_between(
        x_plot[:, idx].ravel(),
        mean_prediction[:, idx] - 1.96 * std_prediction[:, idx],
        mean_prediction[:, idx] + 1.96 * std_prediction[:, idx],
        alpha=0.5,
        label=r"95% confidence ",
    )
plt.xlabel('Time steps')
plt.ylabel('Gene Expression')
plt.legend()
plt.show()
