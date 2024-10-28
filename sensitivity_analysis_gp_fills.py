import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pydmd import HODMD, DMD, SpDMD

from utils_bechtold import (get_dataset_bechtold, clean_dataset, fit_gaussian_process,
                            plot_scatter_data, get_gp_predictions)

"""
loads bechtold dataset, augments data with a GP. then tries to performs sensitivity analysis 
UPDATE: linear model sensitivity analysis is simply the matrix A --> SALib is useless.  
"""

SEED = 167
np.random.seed(SEED)
plot = False


drought_data, water_data = get_dataset_bechtold()

X_train_dr, y_dr, X_finer, y_std_dr = clean_dataset(drought_data)
X_train_wt, y_wt, X_finer, y_std_wt = clean_dataset(water_data)

gp_pred_dr, std_pred_dr = get_gp_predictions(X=X_train_dr, Y=y_dr, Y_std=y_std_dr, X_finer=X_finer)
gp_pred_wt, std_pred_wt = get_gp_predictions(X=X_train_wt, Y=y_wt, Y_std=y_std_wt, X_finer=X_finer)

# update n_genes accordingly
n_genes = gp_pred_dr.shape[0]


# sample from GP with randomicity
noisy_gp_dr = std_pred_dr * 0.75 * np.random.randn(gp_pred_dr.shape[0], gp_pred_dr.shape[1]) + gp_pred_dr
noisy_gp_wt = std_pred_wt * 0.75 * np.random.randn(gp_pred_wt.shape[0], gp_pred_wt.shape[1]) + gp_pred_wt

# snapshot should be of dimension (n_genes x time_steps)
snapshots_dr = noisy_gp_dr
snapshots_wt = noisy_gp_wt
new_time = X_finer[:, 0]

cols_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
color_data = cols_all[0]
color_dmd = cols_all[1]
color_spdmd = cols_all[2]
color_hodmd = cols_all[-1]

n_genes_for_plot = 2
genes_for_plot = sorted(np.random.randint(low=0, high=n_genes+1, size=n_genes_for_plot))

from scipy.linalg import pinv

# test different reduced-order models
ranks = [5, 10, 25]
for r in ranks:
    # test and fit HODMD
    hodmd_dr = HODMD(svd_rank=r, d=12)
    hodmd_dr.fit(snapshots_dr)

    ranks_dr = hodmd_dr.modes.shape[1]
    d_dr = 12

    # compute A operator
    A_hodmd_dr = np.linalg.multi_dot([hodmd_dr.modes, np.diag(hodmd_dr.eigs), pinv(hodmd_dr.modes)])
    # this is dimension (d * internal_svd_rank) x (d * internal_svd_rank), since the hankel matrix takes an
    # internal projection into a smaller space
    Atilde_hodmd_dr = hodmd_dr.operator.as_numpy_array # this is (rank x rank)

    #####################
    # water data
    #####################

    hodmd_wt = HODMD(svd_rank=r, d=12)
    hodmd_wt.fit(snapshots_wt)

    ranks_wt = hodmd_wt.modes.shape[1]
    d_wt = 12

    # compute A operator
    A_hodmd_wt = np.linalg.multi_dot([hodmd_wt.modes, np.diag(hodmd_wt.eigs), pinv(hodmd_wt.modes)])
    # this is dimension (d * internal_svd_rank) x (d * internal_svd_rank), since the hankel matrix takes an
    # internal projection into a smaller space
    Atilde_hodmd_wt = hodmd_wt.operator.as_numpy_array  # this is (rank x rank)

    plt.figure()
    plt.imshow(np.log(np.abs(Atilde_hodmd_dr)))
    plt.yticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks_dr+1)])
    plt.xticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks_dr+1)])
    plt.title('input/output importance (drought)')
    plt.colorbar()

    plt.figure()
    plt.imshow(np.log(np.abs(Atilde_hodmd_wt)))
    plt.yticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks_dr+1)])
    plt.xticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks_dr+1)])
    plt.title('input/output importance (water)')
    plt.colorbar()

############################################################################
# Sensitivity analysis input/output
############################################################################

from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol, delta

# def fx(x, A):
#     """Return y = a + b*x**2."""
#     return (A @ x.T).T
#
#
# var_names =  ['x' + str(i) for i in range(1, ranks_dr+1)]
# problem = ProblemSpec({
#     'num_vars': ranks_dr,
#     'names': ['x' + str(i) for i in range(1, ranks_dr+1)],
#     'bounds': [[-1, 1]]*ranks_dr,
#     'outputs': ['x' + str(i) + '_next' for i in range(1, ranks_dr+1)]
# })
#
#
# # sample
# param_values = saltelli.sample(problem, 2**ranks_dr)
#
# # evaluate
# y_dr = np.array([fx(params, A=Atilde_hodmd_dr) for params in param_values])
#
# # analyse
# sobol_indices = [sobol.analyze(problem, Y) for Y in y_dr.T]
#
# print('sample and sobol analysis')
# (problem.sample_saltelli(2**ranks_dr).evaluate(fx, A=Atilde_hodmd_dr).analyze_sobol(nprocs=2))
#
# print('start analysis delta')
# problem.analyze_delta(num_resamples=5)
# print('analysis finished')
#
# plt.figure()
# deltas = np.zeros((ranks_dr, ranks_dr))
# for idx in range(ranks_dr):
#     deltas[idx,:] = np.log(problem.analysis['x' + str(idx+1) + '_next']['delta'])
#
# plt.imshow(deltas)
# plt.yticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks_dr+1)])
# plt.xticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks_dr+1)])
# plt.title('input/output importance (drought)')
# plt.colorbar()
#
# plt.figure()
# plt.imshow(Atilde_hodmd_dr)
# plt.yticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks_dr+1)])
# plt.xticks(list(range(ranks_dr)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks_dr+1)])
# plt.title('input/output importance (drought)')
# plt.colorbar()
#
#
#
#
# # evaluate
# y_wt = np.array([fx(params, A=Atilde_hodmd_wt) for params in param_values])
#
# # analyse
# sobol_indices = [sobol.analyze(problem, Y) for Y in y_wt.T]
#
# print('sample and sobol analysis')
# (problem.sample_saltelli(2**ranks_dr).evaluate(fx, A=Atilde_hodmd_wt).analyze_sobol(nprocs=2))
#
# print('start analysis delta')
# problem.analyze_delta(num_resamples=5)
# print('analysis finished')
#
# plt.figure()
# deltas = np.zeros((ranks_wt, ranks_wt))
# for idx in range(ranks_wt):
#     deltas[idx,:] = np.log(problem.analysis['x' + str(idx+1) + '_next']['delta'])
#
# plt.imshow(deltas)
# plt.yticks(list(range(ranks_wt)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks_wt+1)])
# plt.xticks(list(range(ranks_wt)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks_wt+1)])
# plt.title('input/output importance (water)')
# plt.colorbar()

plt.show()


