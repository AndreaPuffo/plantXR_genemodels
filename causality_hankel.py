import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pydmd import HODMD, DMD, SpDMD, PiDMD
from pydmd.utils import pseudo_hankel_matrix

from utils_bechtold import (get_dataset_bechtold, clean_dataset, fit_gaussian_process,
                            plot_scatter_data, get_gp_predictions)

"""
test to understand how the Hankel matrix work in pydmd
Hankel matrix simply stacks past measurement to form a bigger measurement matrix 
which should be the basic step for HO-DMD 
finally, use the Hankel matrix into a PI-DMD setting
"""

### test functionality of Hankel transform
# vector becomes x at time t=0 --> y at t=1 --> w at t=2 --> z at t=3
X = np.array([
    ['x1', 'y1', 'w1', 'z1'],
    ['x2', 'y2', 'w2', 'z2'],
    ['x3', 'y3', 'w3', 'z3']
])
hankelx = pseudo_hankel_matrix(X, d=2)
print(hankelx)

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

d = 4
cutoff = 10
phm = pseudo_hankel_matrix(snapshots_dr[:cutoff, :], d=d)
pidmd_ut = PiDMD(manifold="diagonal", manifold_opt=(cutoff*d, cutoff+1), compute_A=True).fit(phm)

plt.imshow(pidmd_ut.A, vmax=pidmd_ut.A.max(), vmin=-pidmd_ut.A.max())
plt.colorbar()
plt.show()


