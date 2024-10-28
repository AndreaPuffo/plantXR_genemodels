import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from pydmd import HODMD, DMD, SpDMD, PiDMD
from pydmd.utils import pseudo_hankel_matrix
from utils_bechtold import get_dataset_bechtold, clean_dataset, fit_gaussian_process, plot_scatter_data

"""
loads the Bechtold dataset, cleans it. uses GP to enrich dataset, 
compare normal DMD vs. SpDMD, PiDMD, HODMD to fit and predict gene expression 
for drought and water dataset 
"""

SEED = 167
np.random.seed(SEED)
plot = False

# load and clean dataset 
drought_data, water_data = get_dataset_bechtold()
X_train_dr, y_dr, X_finer, y_std_dr = clean_dataset(drought_data)
X_train_wt, y_wt, X_finer, y_std_wt = clean_dataset(water_data)
# test parameters 
n_genes = len(y_dr)
plot_steps = X_finer.shape[0]

gp_drought = []  # stores all GPs for drought data
gp_water = []
idx_to_remove = []  # stores idx where dataset is not usable
gp_pred_dr, std_pred_dr = np.zeros((n_genes, plot_steps)), np.zeros((n_genes, plot_steps))
gp_pred_wt, std_pred_wt = np.zeros((n_genes, plot_steps)), np.zeros((n_genes, plot_steps))

for idx_gene in range(n_genes):

    # gaussian process for each gene
    gp_drought += [fit_gaussian_process(X_train=X_train_dr[idx_gene], y_train=y_dr[idx_gene], y_std=y_std_dr[idx_gene])]
    gp_water += [fit_gaussian_process(X_train=X_train_wt[idx_gene], y_train=y_wt[idx_gene], y_std=y_std_wt[idx_gene])]

    # use the newly found GP to predict a drought/watered dataset
    if gp_drought[-1] is not None and gp_water[-1] is not None:
        mean_prediction_dr, std_prediction_dr = gp_drought[-1].predict(X_finer, return_std=True)
        mean_prediction_wt, std_prediction_wt = gp_water[-1].predict(X_finer, return_std=True)
        # store results
        gp_pred_dr[idx_gene,:] = mean_prediction_dr
        std_pred_dr[idx_gene, :] = std_prediction_dr
        gp_pred_wt[idx_gene, :] = mean_prediction_wt
        std_pred_wt[idx_gene, :] = std_prediction_wt
    else:
        idx_to_remove += [idx_gene]

# remove zero rows, corresponding to completely empty entry in dataset
gp_pred_dr = np.delete(gp_pred_dr, idx_to_remove, axis=0)
std_pred_dr = np.delete(std_pred_dr, idx_to_remove, axis=0)
gp_pred_wt = np.delete(gp_pred_wt, idx_to_remove, axis=0)
std_pred_wt = np.delete(std_pred_wt, idx_to_remove, axis=0)

# update n_genes accordingly
n_genes = gp_pred_dr.shape[0]

# sample from GP with randomicity
noisy_gp_dr = std_pred_dr * 0.75 * np.random.randn(gp_pred_dr.shape[0], gp_pred_dr.shape[1]) + gp_pred_dr
noisy_gp_wt = std_pred_wt * 0.75 * np.random.randn(gp_pred_wt.shape[0], gp_pred_wt.shape[1]) + gp_pred_wt

# snapshot should be of dimension (n_genes x time_steps)
snapshots_dr = noisy_gp_dr
snapshots_wt = noisy_gp_wt
new_time = X_finer[:, 0]

# just increase these lists to check what combination works better
ranks = [12, 15, 24]  # rank is the dimension of the reduced order system 
ds = [5, 12, 15]  # ds is the delay index to use in the HODMD 
res_dmd = np.zeros((len(ranks), 2))
res_hodmd = np.zeros((len(ranks), 2))
res_spdmd = np.zeros((len(ranks), 2))
cols_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
color_data = cols_all[0]
color_dmd = cols_all[1]
color_spdmd = cols_all[2]
color_pidmd = cols_all[3]
color_hodmd = cols_all[4]

n_genes_for_plot = 2
genes_for_plot = sorted(np.random.randint(low=0, high=n_genes+1, size=n_genes_for_plot))
# fitting dor all combinations of ranks and ds 
for t in range(len(ranks)):
    dmd_dr = DMD(svd_rank=ranks[t])
    dmd_dr.fit(snapshots_dr)

    USV = np.linalg.multi_dot([dmd_dr.modes, np.diag(dmd_dr.eigs), pinv(dmd_dr.modes)])

    spdmd_dr = SpDMD(svd_rank=ranks[t])
    spdmd_dr.fit(snapshots_dr)

    hodmd_dr = HODMD(svd_rank=ranks[t], d=ds[t])
    hodmd_dr.fit(snapshots_dr)
    # hodmd is NOT causal, so use physics-informed DMD

    phm = pseudo_hankel_matrix(snapshots_dr, d=ds[t])
    pidmd_dr = PiDMD(svd_rank=ranks[t], manifold='diagonal', manifold_opt=(n_genes * ds[t], n_genes + 1),
                     compute_A=True)
    pidmd_dr.fit(phm)

    # compute A operator
    A_hodmd_dr = np.linalg.multi_dot([hodmd_dr.modes, np.diag(hodmd_dr.eigs), pinv(hodmd_dr.modes)])
    # this is dimension (d * internal_svd_rank) x (d * internal_svd_rank), since the hankel matrix takes an
    # internal projection into a smaller space
    Atilde_hodmd_dr = hodmd_dr.operator.as_numpy_array # this is (rank x rank)
    Atilde_pidmd_dr = pidmd_dr.operator.as_numpy_array  # this is (rank x rank)

    print(f'Number of modes: {hodmd_dr.modes.shape[1]}')
    print(f'MSE DMD: {np.mean(np.linalg.norm(snapshots_dr - dmd_dr.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE SpDMD: {np.mean(np.linalg.norm(snapshots_dr - spdmd_dr.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE HODMD: {np.mean(np.linalg.norm(snapshots_dr - hodmd_dr.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE PiDMD: {np.mean(np.linalg.norm(snapshots_dr[:,-(new_time.shape[0]-ds[t]+1):] - pidmd_dr.reconstructed_data[-n_genes:,:], ord=2, axis=1))}')


    res_dmd[t, 0] = np.mean(np.linalg.norm(snapshots_dr - dmd_dr.reconstructed_data, ord=2, axis=1))
    res_spdmd[t, 0] = np.mean(np.linalg.norm(snapshots_dr - spdmd_dr.reconstructed_data, ord=2, axis=1))
    res_hodmd[t, 0] = np.mean(np.linalg.norm(snapshots_dr - hodmd_dr.reconstructed_data, ord=2, axis=1))

    #####################
    # water data
    #####################

    dmd_wt = DMD(svd_rank=ranks[t])
    dmd_wt.fit(snapshots_wt)

    spdmd_wt = SpDMD(svd_rank=ranks[t])
    spdmd_wt.fit(snapshots_wt)

    hodmd_wt = HODMD(svd_rank=ranks[t], d=ds[t])
    hodmd_wt.fit(snapshots_wt)

    phm = pseudo_hankel_matrix(snapshots_wt, d=ds[t])
    pidmd_wt = PiDMD(svd_rank=ranks[t], manifold='diagonal', manifold_opt=(n_genes * ds[t], n_genes + 1),
                     compute_A=True)
    pidmd_wt.fit(phm)

    # compute A operator
    Atilde_hodmd_wt = hodmd_wt.operator.as_numpy_array

    print(f'Number of modes: {hodmd_wt.modes.shape[1]}')
    print(f'MSE DMD: {np.mean(np.linalg.norm(snapshots_wt - dmd_wt.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE SpDMD: {np.mean(np.linalg.norm(snapshots_wt - spdmd_wt.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE HODMD: {np.mean(np.linalg.norm(snapshots_wt - hodmd_wt.reconstructed_data, ord=2, axis=1))}')
    print(f'MSE PiDMD: {np.mean(np.linalg.norm(snapshots_wt[:,-(new_time.shape[0]-ds[t]+1):] - pidmd_wt.reconstructed_data[-n_genes:,:], ord=2, axis=1))}')


    res_dmd[t, 1] = np.mean(np.linalg.norm(snapshots_wt - dmd_wt.reconstructed_data, ord=2, axis=1))
    res_spdmd[t, 1] = np.mean(np.linalg.norm(snapshots_wt - spdmd_wt.reconstructed_data, ord=2, axis=1))
    res_hodmd[t, 1] = np.mean(np.linalg.norm(snapshots_wt - hodmd_wt.reconstructed_data, ord=2, axis=1))

    #####################
    # plots
    #####################

    plt.figure()
    plot_scatter_data(time=new_time, data=snapshots_dr, indices=genes_for_plot,
                      color_data=color_data, label='Data', scatter=True)
    plot_scatter_data(time=new_time, data=dmd_dr.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_dmd, label='DMD', scatter=False)
    plot_scatter_data(time=new_time, data=spdmd_dr.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_spdmd, label='SpDMD', scatter=False)
    plot_scatter_data(time=new_time, data=hodmd_dr.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_hodmd, label='HoDMD', scatter=False)
    plot_scatter_data(time=new_time[new_time.shape[0]-(new_time.shape[0]-ds[t]+1):],
                      data=pidmd_dr.reconstructed_data[-n_genes:,:], indices=genes_for_plot,
                        color_data=color_pidmd, label='PiDMD', scatter=False)

    plt.legend()
    plt.grid()
    plt.xlabel('Hours')
    plt.ylabel('Gene Expression (normalised)')
    if ranks[t] != 0:
        plt.title(f'Drought / DMD rank: {ranks[t]}, ds: {ds[t]}')
    else:
        plt.title(f'Drought / DMD ranks: {dmd_dr.modes.shape[1], spdmd_dr.modes.shape[1], hodmd_dr.modes.shape[1]}, '
                  f'ds: {ds[t]}')

    plt.figure()
    plot_scatter_data(time=new_time, data=snapshots_wt, indices=genes_for_plot,
                      color_data=color_data, label='Data', scatter=True)
    plot_scatter_data(time=new_time, data=dmd_wt.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_dmd, label='DMD', scatter=False)
    plot_scatter_data(time=new_time, data=spdmd_wt.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_spdmd, label='SpDMD', scatter=False)
    plot_scatter_data(time=new_time, data=hodmd_wt.reconstructed_data.real, indices=genes_for_plot,
                      color_data=color_hodmd, label='HoDMD', scatter=False)
    plot_scatter_data(time=new_time[new_time.shape[0] - (new_time.shape[0] - ds[t] + 1):],
                      data=pidmd_wt.reconstructed_data[-n_genes:, :], indices=genes_for_plot,
                      color_data=color_pidmd, label='PiDMD', scatter=False)
    plt.legend()
    plt.grid()
    plt.xlabel('Hours')
    plt.ylabel('Gene Expression (normalised)')
    plt.title('Water data')
    if ranks[t] != 0:
        plt.title(f'Water / DMD rank: {ranks[t]}, ds: {ds[t]}')
    else:
        plt.title(f'Water / DMD rank: {dmd_wt.modes.shape[1], spdmd_wt.modes.shape[1], hodmd_wt.modes.shape[1]}, '
                  f'ds: {ds[t]}')

plt.figure()
plt.plot(res_dmd[:, 0], '-.', c=color_dmd, label='DMD drought')
plt.plot(res_dmd[:, 1], '--', c=color_dmd, label='DMD water')
plt.plot(res_spdmd[:, 0], '-.', c=color_spdmd, label='SpDMD drought')
plt.plot(res_spdmd[:, 1], '--', c=color_spdmd, label='SpDMD water')
plt.plot(res_hodmd[:, 0], '-.', c=color_hodmd, label='HODMD drought')
plt.plot(res_hodmd[:, 1], '--', c=color_hodmd, label='HODMD water')
plt.legend()
plt.grid()
plt.xlabel('DMD Rank')
plt.xticks(list(range(len(ranks))), [str(r) for r in ranks])
plt.ylabel('MSE')


plt.show()



