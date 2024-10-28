"""
tutorial from the pydmd github page
https://pydmd.github.io/PyDMD/tutorial1dmd.html
https://pydmd.github.io/PyDMD/tutorial2dmd.html
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydmd import DMD
from pydmd.plotter import plot_eigs

def f1(x,t):
    return 1./np.cosh(x+3)*np.exp(1 + 0*2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(1 - 2.8*t)

n_points = 1
time_points = 95
x = np.linspace(-5, 5, n_points)
t = np.linspace(0, 4*np.pi, time_points)

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2


titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data = [X1, X2, X]

fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131,134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()


# Actual DMD
dmd = DMD(svd_rank=0)
dmd.fit(X.T)

for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

plot_eigs(dmd=dmd, show_axes=True, show_unit_circle=True)

for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')

plt.figure()
for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')

fig = plt.figure(figsize=(17, 6))

for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)

plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()


plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
fig = plt.colorbar()


# prediction
x = np.linspace(-5, 5, n_points)
t = np.linspace(0, 4*np.pi, time_points)

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

y_predict_dmd = dmd.predict(X[:-1, :].T)
plt.figure()
plt.pcolor(xgrid[1:, :], tgrid[1:, :], abs((X[1:, :]-y_predict_dmd.T).real))
fig = plt.colorbar()
plt.title('prediction error')

plt.show()

