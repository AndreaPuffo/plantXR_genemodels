"""
from pydmd tutorials page
https://pydmd.github.io/PyDMD/tutorial6hodmd.html
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from pydmd import HODMD
from pydmd.plotter import plot_eigs

def myfunc(x):
    return np.cos(x)*np.sin(np.cos(x)) + np.cos(x*.2)

# Because we trust in the DMD power, we add a bit of noise and we plot our function:

x = np.linspace(0, 10, 64)
y = myfunc(x)
snapshots = np.atleast_2d(y)
plt.plot(x, snapshots[0, :], '.')

hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(snapshots)

# Despite the arrangement, the shape of the reconstructed data is the same of the original input.
# hodmd.reconstructed_data.shape

# As always, we take a look at the eigenvalues to check the stability of the system.

plot_eigs(hodmd)

hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = x[1] - x[0]
hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = x[0]
hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = x[-1]

plt.plot(hodmd.original_timesteps, snapshots[0,:], '.', label='snapshots')
plt.plot(hodmd.original_timesteps, y, '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()

plt.figure(figsize=(7,5))
plt.plot(hodmd.dynamics.T)
plt.xlabel('x')
plt.ylabel('Modes')
plt.grid()

hodmd.dmd_time['tend'] = 50

fig = plt.figure(figsize=(15, 5))
plt.plot(hodmd.original_timesteps, snapshots[0,:], '.', label='snapshots')
plt.plot(np.linspace(0, 50, 128), myfunc(np.linspace(0, 50, 128)), '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()


noise_range = [.01, .05, .1, .2]
fig = plt.figure(figsize=(15, 10))
future = 20

for id_plot, i in enumerate(noise_range, start=1):
    snapshots = y + np.random.uniform(-i, i, size=y.shape)
    hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(np.atleast_2d(snapshots))
    hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = x[1] - x[0]
    hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = x[0]
    hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = x[-1]
    hodmd.dmd_time['tend'] = 20

    plt.subplot(2, 2, id_plot)
    plt.plot(hodmd.original_timesteps, snapshots, '.', label='snapshots')
    plt.plot(np.linspace(0, future, 128), myfunc(np.linspace(0, future, 128)), '-', label='original function')
    plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
    plt.legend()
    plt.title('Noise [{} - {}]'.format(-i, i))
plt.show()

# only last plot
plt.figure(figsize=(7,5))
plt.plot(hodmd.original_timesteps, snapshots, '.', label='Samples')
plt.plot(np.linspace(0, future, 128), myfunc(np.linspace(0, future, 128)), '-', label='Original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
# plt.title('Noise [{} - {}]'.format(-i, i))



plt.show()