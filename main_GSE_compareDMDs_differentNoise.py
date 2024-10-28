import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pydmd import HODMD, DMD, SpDMD
"""
loads the GSE dataset, cleans it. uses GP to enrich dataset, 
compare normal DMD vs.  HODMD to fit and predict gene expression. 
"""
SEED = 167
np.random.seed(SEED)
plot = False
# load dataset and fix test params
data = pd.read_csv('datasets/gse5612_redux.txt', delimiter=';', index_col=0)
n_genes = 62
n_genes_for_plot = 1
time_points = 13
y_main = data.values[0:n_genes,:].T
# clean dataset
y_main = y_main / np.max(y_main, axis=0)
# zero-mean and std   ---> standardization doesnt work well!
# y_main = (y_main - y_main.mean(axis=0)) / y_main.std(axis=0)
X_train = np.linspace(start=0, stop=time_points*4, num=time_points).reshape(-1, 1)
X_train = np.tile(X_train, (n_genes, ))
time = X_train

# different values of noise variance
noise_stds = [0.05, 0.1]
res_dmd = np.zeros((2, 6))
res_hodmd = np.zeros((2, 6))
res_emd_hodmd = np.zeros((2, 6))
res_spdmd = np.zeros((2, 6))
# fit for noise variance
for idx_noise, n in enumerate(noise_stds):

    print('-'*80)
    print(f'Noise standard deviation: {n}')

    kernel = 1 * RBF(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=n**2,
                                                n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_main)

    # get mean prediction and std from the GP
    x_plot = np.linspace(start = 0, stop = time_points*4, num = 100).reshape(-1, 1)
    x_plot = np.tile(x_plot, (n_genes, ))
    mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

    cols_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    cols = cols_all[:n_genes_for_plot]

    if plot:
        fig, ax = plt.subplots()
        default_cycler = cycler(color=cols)
        ax.set_prop_cycle(default_cycler)

        for idx in range(n_genes_for_plot):
            if idx == 0:
                plt.errorbar(
                    X_train[:, idx],
                    y_main[:, idx],
                    n,
                    linestyle="None",
                    # color=cols[idx],
                    marker=".",
                    markersize=10,
                    label="Observations",
                )
            else:
                plt.errorbar(
                    X_train[:, idx],
                    y_main[:, idx],
                    n,
                    linestyle="None",
                    # color=cols[idx],
                    marker=".",
                    markersize=10,
                )


        plt.plot(x_plot[:, :n_genes_for_plot], mean_prediction[:, :n_genes_for_plot])
        plt.plot([], [], label='Mean prediction')
        for idx in range(n_genes_for_plot):
            plt.fill_between(
                x_plot[:, idx].ravel(),
                mean_prediction[:, idx] - 1.96 * std_prediction[:, idx],
                mean_prediction[:, idx] + 1.96 * std_prediction[:, idx],
                alpha=0.5,
            )
        plt.fill_between([], [], alpha=0.5, label='95% prediction')

        plt.xlabel('Hours')
        plt.ylabel('Gene expression (normalised)')
        plt.legend(loc='best')
        plt.grid()


    ########################################################################
    #  high order dmd
    ########################################################################

    # sample from GP with randomicity
    # noisy_gp = np.random.uniform(low=mean_prediction - 1.96 * std_prediction,
    #                                  high= mean_prediction + 1.96 * std_prediction)
    noisy_gp = std_prediction * 0.75 * np.random.randn(mean_prediction.shape[0], mean_prediction.shape[1]) + mean_prediction
    if plot:
        plt.figure()
        plt.plot(mean_prediction[:, 0] - 1.96 * std_prediction[:, 0], '--', c='k')
        plt.plot(mean_prediction[:, 0] + 1.96 * std_prediction[:, 0], '--', c='k')
        plt.plot([], [], label='GP 95% bounds', c='k')
        plt.plot(noisy_gp[:, 0], label='Synthetic data')
        plt.xlabel('Time [Hours]')
        plt.ylabel('Gene expression (normalised)')
        plt.legend()
        plt.grid()


    snapshots = noisy_gp.T
    new_time = x_plot[:, 0]

    # embedding of data
    s_squared = snapshots**2
    s_cube = snapshots**3
    s_sin = np.sin(snapshots)
    s_cos = np.cos(snapshots)
    s_sigmoid = 1./(1. + np.exp(-snapshots))
    s_exp = np.exp(snapshots)

    snapshots_embedded = np.vstack([snapshots, s_squared, s_cube, s_sin, s_cos, s_sigmoid, s_exp])

    # rank is the reduce-order dimension, NOTA: use 0 for automatic selection of rank
    # ds is the delay index to use for the HODMD
    ranks = [5, 7, 10, 12, 15, 20]
    ds = [10, 10, 10, 10, 10, 10]
    for t in range(len(ranks)):
        dmd = DMD(svd_rank=ranks[t])
        dmd.fit(snapshots)

        spdmd = SpDMD(svd_rank=ranks[t])
        spdmd.fit(snapshots)

        hodmd = HODMD(svd_rank=ranks[t], d=ds[t])
        hodmd.fit(snapshots)
        # compute A operator
        hodmd._Atilde.compute_operator(X=snapshots[:-1, :], Y=snapshots[1:, :])
        Atilde_hodmd = hodmd._Atilde.as_numpy_array

        hodmd_emb = HODMD(svd_rank=ranks[t], d=ds[t])
        hodmd_emb.fit(snapshots_embedded)

        print(f'Number of modes: {hodmd.modes.shape[1]}')
        print(f'MSE DMD: {np.mean(np.linalg.norm(snapshots - dmd.reconstructed_data, ord=2, axis=1))}')
        print(f'MSE SpDMD: {np.mean(np.linalg.norm(snapshots - spdmd.reconstructed_data, ord=2, axis=1))}')
        print(f'MSE HODMD: {np.mean(np.linalg.norm(snapshots - hodmd.reconstructed_data, ord=2, axis=1))}')
        print(f'MSE embedding: {np.mean(np.linalg.norm(snapshots - hodmd_emb.reconstructed_data[:snapshots.shape[0],:], ord=2, axis=1))}')

        res_dmd[idx_noise, t] = np.mean(np.linalg.norm(snapshots - dmd.reconstructed_data, ord=2, axis=1))
        res_spdmd[idx_noise, t] = np.mean(np.linalg.norm(snapshots - spdmd.reconstructed_data, ord=2, axis=1))
        res_hodmd[idx_noise, t] = np.mean(np.linalg.norm(snapshots - hodmd.reconstructed_data, ord=2, axis=1))
        res_emd_hodmd[idx_noise, t] = np.mean(np.linalg.norm(snapshots - hodmd_emb.reconstructed_data[:snapshots.shape[0],:], ord=2, axis=1))


        plt.figure()
        for idx in range(n_genes_for_plot):
            plt.scatter(new_time, snapshots[idx,:], marker='.', c='tab:blue')
        plt.scatter([], [], marker='.', c='tab:blue', label='Data')

        for idx in range(n_genes_for_plot):
            plt.plot(new_time, dmd.reconstructed_data[idx, :].real, '-', c='tab:orange')
        plt.plot([], [], '-', c='tab:orange', label='DMD')

        for idx in range(n_genes_for_plot):
            plt.plot(new_time, spdmd.reconstructed_data[idx,:].real, '-', c='tab:cyan')
        plt.plot([], [], '-', c='tab:cyan', label='SpDMD')

        for idx in range(n_genes_for_plot):
            plt.plot(new_time, hodmd.reconstructed_data[idx,:].real, '-', c='tab:green')
        plt.plot([], [], '-', c='tab:green', label='HODMD')

        for idx in range(n_genes_for_plot):
            plt.plot(new_time, hodmd_emb.reconstructed_data[idx,:].real, '-', c='tab:purple')
        plt.plot([], [], '-', c='tab:purple', label='HODMD embedding')

        plt.legend()
        plt.grid()
        plt.xlabel('Hours')
        plt.ylabel('Gene Expression (normalised)')
        if ranks[t] != 0:
            plt.title(f'DMD rank: {ranks[t]}, ds: {ds[t]}')
        else:
            plt.title(f'DMD rank: {hodmd.modes.shape[1]}, ds: {ds[t]}')

plt.figure()
plt.plot(res_dmd[0,:], '-', c='tab:blue', label='DMD')
plt.plot(res_dmd[1,:], '--', c='tab:blue')
plt.plot(res_spdmd[0,:], '-', c='tab:cyan', label='SpDMD')
plt.plot(res_spdmd[1,:], '--', c='tab:cyan')
plt.plot(res_hodmd[0,:], '-', c='tab:orange', label='HODMD')
plt.plot(res_hodmd[1,:], '--', c='tab:orange')
plt.plot(res_emd_hodmd[0,:], '-', c='tab:purple', label='Emb-HODMD')
plt.plot(res_emd_hodmd[1,:], '--', c='tab:purple')
plt.legend()
plt.grid()
plt.xlabel('DMD Rank')
plt.xticks([0, 1, 2, 3, 4, 5], ['5', '7', '10', '12', '15', '20'])
plt.ylabel('MSE')


plt.show()



