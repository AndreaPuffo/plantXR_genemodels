import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
"""
loads the GSE dataset, cleans it. uses GP to enrich dataset, 
compare normal DMD vs.  HODMD to fit and predict gene expression. 
"""

SEED = 167
np.random.seed(SEED)
# load dataset
data = pd.read_csv('datasets/gse5612_redux.txt', delimiter=';', index_col=0)
# test parameters
n_genes = 50
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

# try GP fit with two noise variance values
noise_stds = [0.05, 0.1]
for n in noise_stds:
    # GP fitting
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

    from pydmd import HODMD, DMD

    # sample from GP with randomicity
    # noisy_gp = np.random.uniform(low=mean_prediction - 1.96 * std_prediction,
    #                                  high= mean_prediction + 1.96 * std_prediction)
    noisy_gp = std_prediction * 0.75 * np.random.randn(mean_prediction.shape[0], mean_prediction.shape[1]) + mean_prediction
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
    # test HO-DMD vs. normal DMD
    ranks = [5, 10, 20, 0]
    ds = [10, 15, 20, 30]
    for t in range(len(ranks)):
        # fit normal DMD
        dmd = DMD(svd_rank=ranks[t])
        dmd.fit(snapshots)
        # fit HODMD
        hodmd = HODMD(svd_rank=ranks[t], d=ds[t])
        hodmd.fit(snapshots)

        print(f'Number of modes: {hodmd.modes.shape[1]}')
        print(np.mean(np.linalg.norm(snapshots - hodmd.reconstructed_data, ord=2, axis=1)))

        plt.figure()
        for idx in range(n_genes_for_plot):
            plt.scatter(new_time, snapshots[idx,:], marker='.', c=cols[idx])
        plt.scatter([], [], marker='.', label='Data')

        for idx in range(n_genes_for_plot):
            plt.plot(new_time, dmd.reconstructed_data[idx,:].real, '--', c=cols[idx])
        plt.plot([], [], '--', label='DMD output')
        plt.legend()
        plt.grid()
        plt.xlabel('Hours')
        plt.ylabel('Gene Expression (normalised)')



plt.show()



