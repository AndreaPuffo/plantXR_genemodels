# data from Circadian autonomy and rhythmic precision of the Arabidopsis female reproductive organ
# by   Masaaki Okada, Zhiyuan Yang, Paloma Mas
"""
this script loads dataset, and tries some regressions (linear, poly) to fit the data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('mmc_data.csv', delimiter=';')
arr = data.to_numpy(dtype=np.float32)
m = arr.mean(axis=1)
zero_arr = (arr.T - m) /  arr.var(axis=1)

to_remove = []
for c in range(zero_arr.shape[1]):
    if (zero_arr[:, c] == 0).all():
        # remove column
        to_remove += [c]


zero_arr = np.delete(zero_arr, to_remove, 1)

next = zero_arr[1:, :]
prec = zero_arr[:-1, :]

# predictor = ((prec.T @ prec)**-1) @ prec.T @ next
#
# yhat = predictor @ prec
# err = next - yhat

feature_num = 3
poly = PolynomialFeatures(degree=2, include_bias=True)
poly_features = poly.fit_transform(prec[:, 0:feature_num])
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y=next[:, 0:feature_num])

y_predicted = poly_reg_model.predict(poly_features)

c = ['blue', 'orange', 'green']
x_plot = np.arange(start=3, stop=11*4+3, step=4)
for i in range(feature_num):
    plt.plot(x_plot,y_predicted[:, i], label='predicted', linestyle='dashed', color=c[i])
    plt.plot(x_plot, next[:, i], label='data', color=c[i])
plt.legend()
plt.grid()
plt.ylabel('Expression (std values)')
plt.xlabel('Time [hr]')
plt.show()