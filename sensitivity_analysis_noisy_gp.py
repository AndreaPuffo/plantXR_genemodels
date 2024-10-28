import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


"""
loads bechtold dataset, augments data with a GP. then tries to performs sensitivity analysis with SALib 
by perturbing the values of the matrix A to see how the output changes. 
NOTA: takes a long time, not very useful (only a linear model) 
"""

SEED = 167
np.random.seed(SEED)
plot = False

# load dataset
data = pd.read_csv('datasets/gse5612_redux.txt', delimiter=';', index_col=0)
n_genes = 62
n_genes_for_plot = 1
time_points = 13
y_main = data.values[0:n_genes,:].T

y_main = y_main / np.max(y_main, axis=0)
# zero-mean and std   ---> standardization doesnt work well!
# y_main = (y_main - y_main.mean(axis=0)) / y_main.std(axis=0)
X_train = np.linspace(start=0, stop=time_points*4, num=time_points).reshape(-1, 1)
X_train = np.tile(X_train, (n_genes, ))
time = X_train


noise_std = 0.05

print('-'*80)
print(f'Noise standard deviation: {noise_std}')
# fit GP for all gene expression data
kernel = 1 * RBF(length_scale=1.0,
                 length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2,
                                            n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_main)

# get mean prediction and std from the GP
x_plot = np.linspace(start = 0, stop = time_points*4, num = 100).reshape(-1, 1)
x_plot = np.tile(x_plot, (n_genes, ))
mean_prediction, std_prediction = gaussian_process.predict(x_plot, return_std=True)

cols_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
cols = cols_all[:n_genes_for_plot]

if plot:
    fig, ax = plt.subplots()
    default_cycler = cycler(color=cols)
    ax.set_prop_cycle(default_cycler)

    for idx in range(n_genes_for_plot):
        if idx == 0:
            plt.errorbar(
                X_train[:, idx],
                y_main[:, idx],
                noise_std,
                linestyle="None",
                # color=cols[idx],
                marker=".",
                markersize=10,
                label="Observations",
            )
        else:
            plt.errorbar(
                X_train[:, idx],
                y_main[:, idx],
                noise_std,
                linestyle="None",
                # color=cols[idx],
                marker=".",
                markersize=10,
            )


    plt.plot(x_plot[:, :n_genes_for_plot], mean_prediction[:, :n_genes_for_plot])
    plt.plot([], [], label='Mean prediction')
    for idx in range(n_genes_for_plot):
        plt.fill_between(
            x_plot[:, idx].ravel(),
            mean_prediction[:, idx] - 1.96 * std_prediction[:, idx],
            mean_prediction[:, idx] + 1.96 * std_prediction[:, idx],
            alpha=0.5,
        )
    plt.fill_between([], [], alpha=0.5, label='95% prediction')

    plt.xlabel('Hours')
    plt.ylabel('Gene expression (normalised)')
    plt.legend(loc='best')
    plt.grid()


########################################################################
#  high order dmd
########################################################################

from pydmd import HODMD
from pydmd import DMD

# sample from GP with randomicity
# noisy_gp = np.random.uniform(low=mean_prediction - 1.96 * std_prediction,
#                                  high= mean_prediction + 1.96 * std_prediction)
noisy_gp = std_prediction * 0.75 * np.random.randn(mean_prediction.shape[0], mean_prediction.shape[1]) + mean_prediction
if plot:
    plt.figure()
    plt.plot(mean_prediction[:, 0] - 1.96 * std_prediction[:, 0], '--', c='k')
    plt.plot(mean_prediction[:, 0] + 1.96 * std_prediction[:, 0], '--', c='k')
    plt.plot([], [], label='GP 95% bounds', c='k')
    plt.plot(noisy_gp[:, 0], label='Synthetic data')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Gene expression (normalised)')
    plt.legend()
    plt.grid()


snapshots = noisy_gp.T
new_time = x_plot[:, 0]

# parameters for DMD
ranks = 12
ds = 10

dmd = DMD(svd_rank=ranks)
dmd.fit(snapshots)
Atilde_dmd = dmd._Atilde.as_numpy_array

hodmd = HODMD(svd_rank=ranks, d=ds)
hodmd.fit(snapshots)

Atilde_hodmd_first = hodmd.operator.as_numpy_array
# compute A operator
hodmd._Atilde.compute_operator(X=snapshots[:-1, :], Y=snapshots[1:, :])
Atilde_hodmd = hodmd._Atilde.as_numpy_array

print(f'Number of modes: {hodmd.modes.shape[1]}')
print(f'MSE DMD: {np.mean(np.linalg.norm(snapshots - dmd.reconstructed_data, ord=2, axis=1))}')
print(f'MSE HODMD: {np.mean(np.linalg.norm(snapshots - hodmd.reconstructed_data, ord=2, axis=1))}')

plt.figure()
for idx in range(n_genes_for_plot):
    plt.scatter(new_time, snapshots[idx,:], marker='.', c='tab:blue')
plt.scatter([], [], marker='.', c='tab:blue', label='Data')

for idx in range(n_genes_for_plot):
    plt.plot(new_time, dmd.reconstructed_data[idx,:].real, '-', c='tab:orange')
plt.plot([], [], '-', c='tab:orange', label='DMD')

for idx in range(n_genes_for_plot):
    plt.plot(new_time, hodmd.reconstructed_data[idx,:].real, '-', c='tab:green')
plt.plot([], [], '-', c='tab:green', label='HODMD')

plt.legend()
plt.grid()
plt.xlabel('Hours')
plt.ylabel('Gene Expression (normalised)')
plt.title(f'DMD rank: {ranks}, ds: {ds}')


############################################################################
# Sensitivity analysis input/output
############################################################################

from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol, delta

def fx(x, A):
    """Return y = a + b*x**2."""
    return (A @ x.T).T


var_names =  ['x' + str(i) for i in range(1, ranks+1)]
problem = ProblemSpec({
    'num_vars': ranks,
    'names': ['x' + str(i) for i in range(1, ranks+1)],
    'bounds': [[-1, 1]]*ranks,
    'outputs': ['x' + str(i) + '_next' for i in range(1, ranks+1)]
})


# sample
param_values = saltelli.sample(problem, 2**ranks)

# evaluate
y = np.array([fx(params, A=Atilde_hodmd) for params in param_values])

# analyse
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

print('sample and sobol analysis')

(problem.sample_saltelli(2**ranks).evaluate(fx, A=Atilde_hodmd).analyze_sobol(nprocs=2))

# print(problem)

# problem.plot()
# plt.show()
#
print('start analysis delta')

problem.analyze_delta(num_resamples=5)
# problem.plot()
# plt.show()

print('analysis finished')

# fig, ax = plt.subplots(1, ranks)
# for idx in range(ranks):
#     ax[idx].bar(list(range(ranks)), problem.analysis['x' + str(idx+1) + '_next']['delta'])
#     ax[idx].set_xticks(list(range(ranks)), var_names)
#     ax[idx].set_title('x' + str(idx+1) + '_next')

plt.figure()
deltas = np.zeros((ranks, ranks))
for idx in range(ranks):
    deltas[idx,:] = np.log(problem.analysis['x' + str(idx+1) + '_next']['delta'])

plt.imshow(deltas)
plt.yticks(list(range(ranks)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks+1)])
plt.xticks(list(range(ranks)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks+1)])
plt.title('Importance of inputs on outputs')
plt.colorbar()
plt.show()



############################################################################
# Sensitivity analysis matrix A
############################################################################

from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol, delta

def fx(x, A):
    """Return y = a + b*x**2."""
    return (A @ x.T).T


epsi = 0.01
var_names =  ['a' + str(i) + str(j) for i in range(1, ranks+1) for j in range(1, ranks+1)]
bounds = [[Atilde_hodmd[i, j]-epsi, Atilde_hodmd[i,j]+epsi] for i in range(ranks) for j in range(ranks)]
problem = ProblemSpec({
    'num_vars': ranks**2,
    'names': var_names,
    'bounds': bounds,
    'outputs': ['x' + str(i) + '_next' for i in range(1, ranks+1)]
})


# sample
param_values = saltelli.sample(problem, 2**ranks)

# evaluate
y = np.array([fx(params, A=Atilde_hodmd) for params in param_values])

# analyse
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

print('sample and sobol analysis')

(problem.sample_saltelli(2**ranks).evaluate(fx, A=Atilde_hodmd).analyze_sobol(nprocs=2))


#
print('start analysis delta')

problem.analyze_delta(num_resamples=5)
# problem.plot()
# plt.show()

print('analysis finished')

# fig, ax = plt.subplots(1, ranks)
# for idx in range(ranks):
#     ax[idx].bar(list(range(ranks)), problem.analysis['x' + str(idx+1) + '_next']['delta'])
#     ax[idx].set_xticks(list(range(ranks)), var_names)
#     ax[idx].set_title('x' + str(idx+1) + '_next')

plt.figure()
deltas = np.zeros((ranks, ranks))
for idx in range(ranks):
    deltas[idx,:] = np.log(problem.analysis['x' + str(idx+1) + '_next']['delta'])

plt.imshow(deltas)
plt.yticks(list(range(ranks)), [r'$x_{' + str(i) + '}$\'' for i in range(1, ranks+1)])
plt.xticks(list(range(ranks)), [r'$x_{' + str(i) + '}$' for i in range(1, ranks+1)])
plt.title('Importance of inputs on outputs')
plt.colorbar()
plt.show()



