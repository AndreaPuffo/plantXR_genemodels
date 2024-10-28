"""# data from Time-Series Transcriptomics Reveals That AGAMOUS-LIKE22 Affects Primary Metabolism
# and Developmental Processes in Drought-Stressed Arabidopsis
# Ulrike Bechtold, et al.
this script loads the dataset, splits it into drought and watered samples, and fits a GP to the data
considers only one gene (just to see how GP performs)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# load dataset
data = pd.read_csv('../datasets/dataset_timeseries_bechtold.csv', delimiter=';', skiprows=1, index_col=0)

# define dataset params
n_genes = 96
time_points = 13
n_plants = 4  # dataset considers 4 plants --> plot data for 4 plants to see the variability among different plants
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

# plotting for presentations
idx_gene = 5
print(f'Printing data of: {data.index[idx_gene]}')
plt.plot(water_data[idx_gene,:,:])
plt.ylabel('Gene Expression')
plt.xlabel('Time (Days)')
plt.title(f'{data.index[idx_gene]}')

# pick one gene out of the n_genes, just for plotting
gene_idx = 52
gene_name = data.axes[0][gene_idx]

plt.figure()

# GP regression + plots
for scenario in range(2):
    if scenario == 0:
        obj = drought_data[gene_idx, :, :]
        label = '_drought'
    else:
        obj = water_data[gene_idx, :, :]
        label = '_water'
    # compute mean and std considering that there are some Nan in the dataset
    y_main = np.nanmean(obj, axis=1)
    og_std = y_main.std()
    y_main = (y_main - y_main.mean())/og_std
    y_std = np.nanstd(obj, axis=1)/og_std
    y_std = 0.25
    # X is simply the timestep
    X_train = np.atleast_2d(range(13)).T
    X = X_train

    # gaussian process regression
    kernel = 1 * RBF(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=y_std**2,
                                                n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_main)

    # prediction w/ GP
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    plt.scatter(X, y_main, label=r"avg value " + label, linestyle="dotted")
    # for i in range(obj.shape[1]):
    #     plt.scatter(X_train, (obj[:, i]-np.nanmean(obj[:, i]))/og_std, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction " + label)
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence " + label,
    )
plt.legend()
plt.xlabel("Days")
plt.ylabel('Gene: ' + gene_name)
_ = plt.title("Gaussian process regression on noisy dataset")

plt.show()
