import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def get_dataset_bechtold():
    """
    dedicated file to return dataset of Bechtold et al. in a comprehensible form
    :return:
    """
    data = pd.read_csv('datasets/dataset_timeseries_bechtold.csv', delimiter=';', skiprows=1, index_col=0)

    n_genes = 95
    time_points = 13
    n_plants = 4

    drought_col = [col for col in data if col.startswith('Drought')]
    water_col = [col for col in data if col.startswith('Water')]

    # .to_numpy(dtype=np.float32).reshape((96, 13, 4))
    # the first row contains the time signature, which is simply 1, 2, 3... 13; so remove it
    drought_data = data[drought_col].to_numpy()[1:, :]
    water_data = data[water_col].to_numpy()[1:, :]

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

    drought_data = drought_data.astype(np.float32).reshape(n_genes, time_points, n_plants)
    water_data = water_data.astype(np.float32).reshape(n_genes, time_points, n_plants)

    return drought_data, water_data

def clean_dataset(y):
    """
    cleans a dataset from not-numbers entries
    :param y: np.array, dataset
    :return:
    Xtrains, array of timepoints
    ytrains, array of cleaned data samples
    X_finer, array of finer timepoints (for GP)
    y_stds, array of standard deviation of the dataset (again, for GP use)
    """
    n_plants = y.shape[2]
    timesteps = y.shape[1]
    n_genes = y.shape[0]
    """
    # NOTA: this for-loop creates analyses genes *one* at a time. 
    # this is due to the fact that each gene has different "holes" in the dataset
    # for example, gene 1 has only 3 entries (out of 13) with numerical data, 
    # whilst gene 2 has 5 entries; this is way we cannot put it into an array with coherent dimensions. 
    # later on, we will create one GP per gene, to then resample with a uniform step
    # across the whole dataset. 
    # Should be optimised, but don't really know how
    """
    Xtrains = []
    ytrains = []
    y_stds = []
    for idx_gene in range(n_genes):
        y_single = y[idx_gene, :, :]
        y_main = np.nanmean(y_single, axis=1) / np.nanmax(y_single)
        # og_std = y_main.std()
        # y_main = (y_main - y_main.mean())/og_std
        # y_std = np.nanstd(obj, axis=1)/og_std
        y_std = np.maximum(
            np.nanstd(y_single / np.nanmax(y_single), axis=1)[np.invert(np.isnan(y_main))] / n_plants,
            0.001)
        X = np.linspace(start=0, stop=timesteps-1, num=timesteps)[:, None]
        X_train = X[np.invert(np.isnan(y_main))]
        y_main = y_main[np.invert(np.isnan(y_main))][:, None]

        Xtrains.append(X_train)
        ytrains.append(y_main)
        y_stds.append(y_std)

        X_finer = np.linspace(start=0, stop=timesteps-1, num=100)[:, None]

    return Xtrains, ytrains, X_finer, y_stds

def fit_gaussian_process(X_train, y_train, y_std):
    """
    fits a GP to the dataset
    :param X_train: array,
    :param y_train: array,
    :param y_std: array, standard deviation of ach sample point
    :return: fitted gaussian process
    """
    # gaussian process regression
    kernel = 1 * RBF(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=y_std ** 2,
                                                n_restarts_optimizer=9)
    if X_train.shape[0] > 0:
        gaussian_process.fit(X_train, y_train)
    else:
        gaussian_process=None
    return gaussian_process

def plot_scatter_data(time, data, indices=None, color_data=None, label='', scatter=False):
    """
    plot (scatterplot) of data
    :param time: array,
    :param data: array,
    :param indices: list, array
    :param color_data: list of colors
    :param label: plot legend label
    :param scatter: if True, scatterplot; if False, normal plot
    :return:
    """

    if indices is None:
        indices = range(data.shape[0])

    if scatter:
        for idx in indices:
            plt.scatter(time, data[idx,:], marker='.', c=color_data)
        plt.scatter([], [], marker='.', c=color_data, label=label)
    else:
        for idx in indices:
            plt.plot(time, data[idx, :], '-', c=color_data)
        plt.plot([], [], '-', c=color_data, label=label)


def get_gp_predictions(X, Y, Y_std, X_finer):
    """
    returns prediction from a GP fitted on the data passed as arguments
    :param X: array,
    :param Y: array,
    :param Y_std: array, standard deviation of Y data
    :param X_finer: array, finer X axis for finer (time) predictions
    :return: gp predictions with 95% confidence band
    """
    n_genes = len(Y)
    plot_steps = X_finer.shape[0]

    gps = []  # stores all GPs for dataset
    idx_to_remove = []  # stores idx where dataset is not usable
    gp_pred, std_pred = np.zeros((n_genes, plot_steps)), np.zeros((n_genes, plot_steps))

    for idx_gene in range(n_genes):

        # gaussian process for each gene
        gps += [
            fit_gaussian_process(X_train=X[idx_gene], y_train=Y[idx_gene], y_std=Y_std[idx_gene])]

        # use the newly found GP to predict a drought/watered dataset
        if gps[-1] is not None:
            mean_prediction, std_prediction = gps[-1].predict(X_finer, return_std=True)
            # store results
            gp_pred[idx_gene, :] = mean_prediction
            std_pred[idx_gene, :] = std_prediction
        else:
            idx_to_remove += [idx_gene]

    # remove zero rows, corresponding to completely empty entry in dataset
    gp_pred = np.delete(gp_pred, idx_to_remove, axis=0)
    std_pred = np.delete(std_pred, idx_to_remove, axis=0)

    return gp_pred, std_pred