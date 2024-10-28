import pandas as pd
import numpy as np
from cycler import cycler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pydmd import HODMD, PiDMD, SpDMD

SEED = 167
np.random.seed(SEED)

data = pd.read_csv('datasets/gse5612_redux.txt', delimiter=';', index_col=0)

n_genes = 50
n_genes_for_plot = 1

time_points = 13
y_main = data.values[0:n_genes,:].T

y_main = y_main / np.max(y_main, axis=0)
# zero-mean and std   ---> standardization doesnt work well!
# y_main = (y_main - y_main.mean(axis=0)) / y_main.std(axis=0)
X_train = np.linspace(start=0, stop=time_points*4, num=time_points).reshape(-1, 1)
X_train = np.tile(X_train, (n_genes, ))
time = X_train


noise_stds = [0.05]
for n in noise_stds:

    kernel = 1 * RBF(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=n**2,
                                                n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_main)

    # get mean prediction and std from the GP
    x_plot = np.linspace(start = 0, stop = time_points*4, num = 100).reshape(-1, 1)
    x_plot = np.tile(x_plot, (n_genes, ))
    mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    cols = cols[:n_genes_for_plot]

    ########################################################################
    #  high order dmd
    ########################################################################

    # sample from GP with randomicity
    noisy_gp = std_prediction * 0.5 * np.random.randn(mean_prediction.shape[0], mean_prediction.shape[1]) + mean_prediction

    snapshots = noisy_gp.T
    new_time = x_plot[:, 0]
    # rank = 0 for automatic selection of rank
    ranks = [0]
    ds = [10]
    hodmd = HODMD(svd_rank=ranks[0], d=ds[0])
    hodmd.fit(snapshots)
    # pidmd takes X as matrix with dimension: (n_vars x n_times)
    # embedding of data
    s_squared = snapshots ** 2
    s_cube = snapshots ** 3
    s_sin = np.sin(snapshots)
    s_cos = np.cos(snapshots)
    s_sigmoid = 1. / (1. + np.exp(-snapshots))
    s_exp = np.exp(snapshots)

    snapshots_embedded = np.vstack([snapshots, s_squared, s_cube, s_sin, s_cos, s_sigmoid, s_exp])
    """
    manifold options   [NOTA: causal --> upper triangular]    
       TYPE OF PIDMD                            RESULT (x = bad, ok = good)
    - "unitary"                                 semi ok 
    - "uppertriangular"                         x
    - "lowertriangular"                         x
    - "diagonal"                                x
    - "symmetric"                               x
    - "skewsymmetric"           
    - "toeplitz"                                x
    - "hankel"                                  x
    - "circulant"                               x
    - "circulant_unitary"                       x
    - "circulant_symmetric"
    - "circulant_skewsymmetric"                 x
    - "symmetric_tridiagonal"
    - "BC" (block circulant)                    semi ok add manifold_opt=(snapshots.shape[0], 7)
    - "BCTB" (BC with tridiagonal blocks)
    - "BCCB" (BC with circulant blocks)
    - "BCCBunitary" (BCCB and unitary)          x
    - "BCCBsymmetric" (BCCB and symmetric)
    - "BCCBskewsymmetric" (BCCB and skewsymmetric)
    """
    pidmd_ut = PiDMD(manifold="BCCBunitary", manifold_opt=(7, snapshots.shape[0]), compute_A=True).fit(snapshots_embedded)

    ########################################################
    # sparsity promoting dmd
    ########################################################

    spdmd = SpDMD(svd_rank=ranks[0])
    spdmd.fit(snapshots)

    print(f'Number of modes: {hodmd.modes.shape[1]}')
    print(f'Mean Error HODMD: {np.mean(np.linalg.norm(snapshots - hodmd.reconstructed_data, ord=2, axis=1))}')
    print(f'Mean Error piDMD: {np.mean(np.linalg.norm(snapshots - pidmd_ut.reconstructed_data[:snapshots.shape[0], :], ord=2, axis=1))}')
    print(
        f'Mean Error spDMD: {np.mean(np.linalg.norm(snapshots - spdmd.reconstructed_data, ord=2, axis=1))}')

    plt.figure()
    for idx in range(n_genes_for_plot):
        plt.scatter(new_time, snapshots[idx,:], marker='.', c=cols[idx])
    plt.scatter([], [], marker='.', label='Data')

    for idx in range(n_genes_for_plot):
        plt.plot(new_time, hodmd.reconstructed_data[idx,:].real, '--', c=cols[idx])
    plt.plot([], [], '--', label='hoDMD output', c=cols[idx])

    for idx in range(n_genes_for_plot):
        plt.plot(new_time, pidmd_ut.reconstructed_data[idx,:].real, '-.', c='orange')
    plt.plot([], [], '-.', label='piDMD output', c='orange')

    for idx in range(n_genes_for_plot):
        plt.plot(new_time, spdmd.reconstructed_data[idx,:].real, '--', c='tab:green')
    plt.plot([], [], '--', label='spDMD output', c='tab:green')

    plt.legend()
    plt.grid()
    plt.xlabel('Hours')
    plt.ylabel('Gene Expression (normalised)')



plt.show()



