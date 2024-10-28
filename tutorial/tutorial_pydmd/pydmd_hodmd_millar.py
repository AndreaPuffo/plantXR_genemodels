import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from pydmd import HODMD, DMD
"""
loads MIllar dataset, uses DMD and HODMD 
"""


# loads millar dataset
data_ = pd.read_csv('../../datasets/12730555174222_data.LIN_DTR.format.csv', delimiter=',')
n_genes = 50
n_genes_plotting = 3
time_steps = 95

time = np.around(data_.values[:, 0], decimals=1)
min_time, max_time = min(time), max(time)
y_main = data_.values[:, 1:1+n_genes]
og_mean, og_std = y_main.mean(axis=0), y_main.std(axis=0)
y_main = (y_main - og_mean)/og_std

x = time
snapshots = y_main.T
plt.plot(x, snapshots[0, :], '.')

#################################
# DMD
#################################

dmd = DMD(svd_rank=2)
dmd.fit(y_main.T)
# plot_modes_2D(dmd, figsize=(12,5))

# plt.figure()
# fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')

fig = plt.figure(figsize=(9,6))
plt.title('Component reconstructed')
plt.plot(dmd.reconstructed_data.T)



fig = plt.figure()
plt.title('Error actual vs. reconstructed data')
plt.plot(y_main - dmd.reconstructed_data.T)

#################################
# HODMD
#################################

hodmd = HODMD(svd_rank=0, d=30)
hodmd.fit(snapshots)

for idx in range(n_genes_plotting):
    plt.plot(time, snapshots[idx,:], '.', label='snapshots')
    # plt.plot(hodmd.original_timesteps, y, '-', label='original function')
    plt.plot(time, hodmd.reconstructed_data[idx].real, '--', label='DMD output')
plt.legend()

hodmd.original_time['dt'] = 1.5
hodmd.original_time['t0'] = time[0]
hodmd.original_time['tend'] = time[-1]
hodmd.fit(snapshots)

plt.show()




