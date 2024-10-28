import numpy as np
import scipy
import scipy.integrate

from matplotlib import animation
# from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd.plotter import plot_modes_2D, plot_snapshots_2D


def fnc(x1, x2):
    return np.stack([x1 + 0.9*x2,
            x1 - 0.1*x2]).T

n_points = 1
x1 = np.linspace(-3, 3, n_points)
x2 = np.linspace(-3, 3, n_points)
x1grid, x2grid = np.meshgrid(x1, x2)

time_hor = 95
time = np.linspace(0, time_hor, time_hor)

data = [np.stack([x1grid, x2grid]).T]
for t in time:
    data += [fnc(data[-1][:,:,0], data[-1][:,:,1])]
noise = [np.random.normal(0.0, 0.1, size=data[0].shape) for t in time]

snapshots = [d+n for d,n in zip(data, noise)]

fig = plt.figure(figsize=(18,12))
plt.title('First Component')
for id_subplot, snapshot in enumerate(snapshots, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot[:, :, 0].real, vmin=-1, vmax=1)

fig = plt.figure(figsize=(18,12))
for id_subplot, snapshot in enumerate(snapshots, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot[:, :, 1].real, vmin=-1, vmax=1)
plt.title('Second component')


dmd = DMD(svd_rank=0)
dmd.fit(snapshots)
# plot_modes_2D(dmd, figsize=(12,5))

plt.figure()
fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')

fig = plt.figure(figsize=(18,12))
plt.title('First component reconstructed')
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.reshape(data[0].shape)[:,:,0].real, vmin=-1, vmax=1)


fig = plt.figure(figsize=(18,12))
plt.title('Second component reconstructed')
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
    plt.subplot(1, time_hor, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.reshape(data[0].shape)[:,:,1].real, vmin=-1, vmax=1)

print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))
dmd.dmd_time['dt'] *= .5
dmd.dmd_time['tend'] *= 1
print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))

fig = plt.figure()

# dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
#
# frames = [
#     [plt.pcolor(x1grid, x2grid, state.real, vmin=-1, vmax=1)]
#     for state in dmd_states
# ]
#
# ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)

# HTML(ani.to_html5_video())

# compute_integral = scipy.integrate.trapz
#
# original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in snapshots]
# dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
#
# figure = plt.figure(figsize=(18, 5))
# plt.plot(dmd.original_timesteps, original_int, 'bo', label='original snapshots')
# plt.plot(dmd.dmd_timesteps, dmd_int, 'r.', label='dmd states')
# plt.ylabel('Integral')
# plt.xlabel('Time')
# plt.grid()
# leg = plt.legend()
# plt.show()

plt.title('Reconstructed data')
plt.subplot(121)
plt.pcolor(x1grid, x2grid, dmd.reconstructed_data.T.real[0].reshape(data[0].shape)[:,:,0])
plt.colorbar()

plt.subplot(122)
plt.pcolor(x1grid, x2grid, dmd.reconstructed_data.T.real[0].reshape(data[0].shape)[:,:,1])
plt.colorbar()


fig = plt.figure()
plt.title('Error actual vs. reconstructed data')
plt.subplot(121)
plt.pcolor(x1grid, x2grid, (data[0]-dmd.reconstructed_data.T[0].reshape(data[0].shape)).real[:,:, 0])
fig = plt.colorbar()

plt.subplot(122)
plt.pcolor(x1grid, x2grid, (data[0]-dmd.reconstructed_data.T[0].reshape(data[0].shape)).real[:,:, 1])
fig = plt.colorbar()

plt.show()
