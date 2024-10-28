import numpy as np
import scipy
import pandas as pd
import scipy.integrate

from matplotlib import animation
# from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd.plotter import plot_modes_2D
"""
this script loads the GSE dataset, fits the DMD, and computes the MSE with the reconstructed data
"""

# load dataset and test params
data_ = pd.read_csv('gse5612_redux.txt', delimiter=';', index_col=0)
n_genes = 2
time_steps = 13
y_main = data_.values[0:n_genes,:].T
X_train = np.linspace(start=0, stop=time_steps*4, num=time_steps).reshape(-1, 1)
X_train = np.tile(X_train, (n_genes, ))
X = X_train
time = np.linspace(0, 6, 16)
noise = [np.random.normal(0.0, 0.1, size=(1, n_genes)) for t in time]
# create noisy data
snapshots = [d+n for d,n in zip(y_main, noise)]
# fit a DMD
dmd = DMD(svd_rank=0, tlsq_rank=2, exact=True, opt=True)
dmd.fit(snapshots)
plot_modes_2D(dmd, figsize=(12,5))
# plot the eigenvals
fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')

print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))

### increase the datapoints automatically within the DMD
dmd.dmd_time['dt'] *= .25
dmd.dmd_time['tend'] *= 3
print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))

fig = plt.figure()

dmd_states = [state.reshape((1, n_genes)) for state in dmd.reconstructed_data.T]

# compare the original data against the reconstructed data of the DMD
compute_integral = scipy.integrate.trapz
original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in snapshots]
dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
# plots
figure = plt.figure(figsize=(18, 5))
plt.plot(dmd.original_timesteps, original_int, 'bo', label='original snapshots')
plt.plot(dmd.dmd_timesteps, dmd_int, 'r.', label='dmd states')
plt.ylabel('Integral')
plt.xlabel('Time')
plt.grid()
leg = plt.legend()
plt.show()
