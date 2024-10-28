import numpy as np
import scipy
import pandas as pd
import scipy.integrate

from matplotlib import animation
# from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd.plotter import plot_modes_2D, plot_snapshots_2D


data_ = pd.read_csv('../../datasets/gse5612_redux.txt', delimiter=';', index_col=0)

n_genes = 5
time_steps = 13
y_main = data_.values[0:n_genes,:].T
# X_train = np.linspace(start=0, stop=time_steps*4, num=time_steps).reshape(-1, 1)
# X_train = np.tile(X_train, (n_genes, ))
# X = X_train

time = np.linspace(0, time_steps, time_steps)

noise = [np.random.normal(0.0, 0.1, size=(1, n_genes)) for t in time]

snapshots = [d+n for d,n in zip(y_main, noise)]


fig = plt.figure(figsize=(9,6))
plt.title('Components')
plt.plot(np.vstack(snapshots))



dmd = DMD(svd_rank=0)
dmd.fit(snapshots)
# plot_modes_2D(dmd, figsize=(12,5))

plt.figure()
fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')

fig = plt.figure(figsize=(9,6))
plt.title('Component reconstructed')
plt.plot(dmd.reconstructed_data.T)


# print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))
# dmd.dmd_time['dt'] *= .5
# dmd.dmd_time['tend'] *= 1
# print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))


fig = plt.figure()
plt.title('Error actual vs. reconstructed data')
plt.plot(y_main - dmd.reconstructed_data.T)

plt.show()
