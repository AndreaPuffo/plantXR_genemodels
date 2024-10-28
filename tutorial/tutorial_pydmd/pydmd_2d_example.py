import numpy as np
import scipy
import scipy.integrate

from matplotlib import animation
# from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd.plotter import plot_modes_2D

n_points = 10
x1 = np.linspace(-3, 3, n_points)
x2 = np.linspace(-3, 3, n_points)
x1grid, x2grid = np.meshgrid(x1, x2)

time_hor = 12
time = np.linspace(0, 1, time_hor)

data = [2/np.cosh(x1grid)/np.cosh(x2grid)*(1 + 0*1.2j**-t) for t in time]
noise = [np.random.normal(0.0, 0.4, size=x1grid.shape) for t in time]

snapshots = [d+n for d,n in zip(data, noise)]

fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(snapshots, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.real, vmin=-1, vmax=1)

dmd = DMD(svd_rank=1, tlsq_rank=2, exact=True, opt=True)
dmd.fit(snapshots)
plot_modes_2D(dmd, figsize=(12,5))

fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')

fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)

print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))
dmd.dmd_time['dt'] *= .5
dmd.dmd_time['tend'] *= 1
print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))

fig = plt.figure()

plt.subplot(121)
plt.pcolor(x1grid, x2grid, dmd.reconstructed_data.T.real[0].reshape((n_points, n_points)))
plt.colorbar()

plt.subplot(122)
plt.pcolor(x1grid, x2grid, (data[0]-dmd.reconstructed_data.T[0].reshape((n_points, n_points))).real)
fig = plt.colorbar()

plt.show()
