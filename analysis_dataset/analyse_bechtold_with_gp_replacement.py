"""# data from Time-Series Transcriptomics Reveals That AGAMOUS-LIKE22 Affects Primary Metabolism
# and Developmental Processes in Drought-Stressed Arabidopsis
# Ulrike Bechtold, et al.
this script loads the dataset, splits it into drought and watered samples, and fits a GP to the data
considers *all* genes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def plot_genes_and_predictions(X, y, mean_prediction, std_prediction, X_finer=None,
                               label='', color='blue', gene_name='unknown'):
    if X_finer is None:
        X_finer = np.linspace(start=X[0], stop=X[-1], num=len(mean_prediction))[:,None]

    plt.scatter(X, y, label=r"avg value " + label, linestyle="dotted",
                c=color)
    # for i in range(obj.shape[1]):
    #     plt.scatter(X_train, (obj[:, i]-np.nanmean(obj[:, i]))/og_std, label="Observations")
    plt.plot(X_finer, mean_prediction, label="Mean prediction " + label,
             c=color)
    plt.fill_between(
        X_finer.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence " + label,
        color=color
    )
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel('Gene: ' + gene_name)
    _ = plt.title("Gaussian process regression on noisy dataset")
    plt.grid()

# load dataset
data = pd.read_csv('../datasets/dataset_timeseries_bechtold.csv', delimiter=';', skiprows=1, index_col=0)
# define dataset params
n_genes = 96
time_points = 13
n_plants = 4
# divide dataset into drought samples and well-watered samples
drought_col = [col for col in data if col.startswith('Drought')]
water_col = [col for col in data if col.startswith('Water')]
drought_data = data[drought_col].to_numpy()
water_data = data[water_col].to_numpy()

# clean the dataset: if there's an empty entry, put NaN
for r in range(drought_data.shape[0]):
    for c in range(drought_data.shape[1]):
        try:
            float(drought_data[r, c])
        except:
            drought_data[r, c] = 'nan'

        try:
            float(water_data[r, c])
        except:
            water_data[r, c] = 'nan'

# finally, reshape the dataset
drought_data = drought_data.astype(np.float32).reshape(n_genes, time_points, n_plants)
water_data = water_data.astype(np.float32).reshape(n_genes, time_points, n_plants)


# plot colors

gp_drought = []
gp_water = []
n_genes_for_plot = n_genes //7
which_genes_to_plot = sorted(np.random.randint(low=1, high=n_genes+1, size=n_genes_for_plot))
data_drought = []
data_water = []
prediction_drought = []
prediction_water = []

# GP regression + plots
for gene_idx in range(1, n_genes):
    gene_name = data.axes[0][gene_idx]

    for scenario in range(2):
        if scenario == 0:
            obj = drought_data[gene_idx, :, :]
            label = '_drought'
        else:
            obj = water_data[gene_idx, :, :]
            label = '_water'
        y_main = np.nanmean(obj, axis=1)/np.nanmax(obj)
        # og_std = y_main.std()
        # y_main = (y_main - y_main.mean())/og_std
        # y_std = np.nanstd(obj, axis=1)/og_std
        y_std = np.maximum(
            np.nanstd(obj/np.nanmax(obj), axis=1)[np.invert(np.isnan(y_main))]/n_plants,
            0.001)
        X = np.linspace(start=0, stop=12, num=13)[:,None]
        X_train = X[np.invert(np.isnan(y_main))]
        # consider only not-NaN values (some genes are practically only NaNs)
        y_main = y_main[np.invert(np.isnan(y_main))][:,None]
        X_finer = np.linspace(start=0, stop=12, num=100)[:,None]

        if y_main.shape[0] == 0:
            print(f'Dataset contains only NaN')
            gaussian_process = None
        else:
            # gaussian process regression
            kernel = 1 * RBF(length_scale=1.0,
                             length_scale_bounds=(1e-2, 1e2))
            gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=y_std**2,
                                                        n_restarts_optimizer=9)
            gaussian_process.fit(X_train, y_main)

        # save GPs and data
        if scenario == 0:
            gp_drought.append(gaussian_process)
            data_drought.append((X_train, y_main))
        else:
            gp_water.append(gaussian_process)
            data_water.append((X_train, y_main))


colors_drought_water = ['tab:red', 'tab:blue']
# for all genes, compute a GP prediction and plot
for idx_gene_to_plot in which_genes_to_plot:
    gene_name = data.axes[0][idx_gene_to_plot]
    data_dr, data_wt = data_drought[idx_gene_to_plot], data_water[idx_gene_to_plot]
    X_finer = np.linspace(start=0, stop=12, num=100)[:, None]

    if gp_drought[idx_gene_to_plot] is not None and gp_water[idx_gene_to_plot] is not None:
        plt.figure()
        mean_prediction_dr, std_prediction_dr = gp_drought[idx_gene_to_plot].predict(X_finer, return_std=True)
        mean_prediction_wt, std_prediction_wt = gp_water[idx_gene_to_plot].predict(X_finer, return_std=True)
        plot_genes_and_predictions(data_dr[0], data_dr[1],
                                   mean_prediction=mean_prediction_dr, std_prediction=std_prediction_dr,
                                   X_finer=X_finer, label='_drought', color='tab:red', gene_name=gene_name)
        plot_genes_and_predictions(data_wt[0], data_wt[1],
                                   mean_prediction=mean_prediction_wt, std_prediction=std_prediction_wt,
                                   X_finer=X_finer, label='_water', color='tab:blue', gene_name=gene_name)



plt.show()
