import numpy as np
import matplotlib.pyplot as plt

from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol, delta


def fx(x, A):
    """Return y = a + b*x**2."""
    return (A @ x.T).T


problem = ProblemSpec({
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[0, 1]]*2,
    'outputs': ['x1_next', 'x2_next']
})


# sample
param_values = saltelli.sample(problem, 2**6)


# evaluate
A=np.array([[0.5, 0.9], [0.03, 0.2]])
y = np.array([fx(params, A=A) for params in param_values])

# analyse
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]


(problem.sample_saltelli(2**6).evaluate(fx, A=A).analyze_sobol(nprocs=2))

print(problem)

# problem.plot()
# plt.show()
#
problem.analyze_delta(num_resamples=5)
# problem.plot()
# plt.show()
#
# fig, ax = plt.subplots(1, 2)
# ax[0].bar([0, 1], sobol_indices[0]['S1'])
# ax[0].set_xticks([0, 1], ['x1', 'x2'])
# ax[0].set_title('x1_next')
# ax[1].bar([0, 1], sobol_indices[1]['S1'])
# ax[1].set_xticks([0, 1], ['x1', 'x2'])
# ax[1].set_title('x2_next')
# plt.show()


####################################################################
#  parameters
####################################################################



def fx_params(x, A):
    """Return y = a + b*x**2."""
    A = A.reshape((2,2))
    return (A @ x.T).T


problem = ProblemSpec({
    'num_vars': 4,
    'names': ['a11', 'a12', 'a21', 'a22'],
    'bounds': [[0, 1]]*4,
    'outputs': ['x1_n', 'x2_n']
})


# sample
param_values = saltelli.sample(problem, 2**4)


# evaluate
num = 50
x1 = np.linspace(-1, 1, num)
x2 = np.linspace(-1, 1, num)
X = np.array(np.meshgrid(x1, x2)).reshape(num**2, 2)
y = np.array([fx_params(X, params) for params in param_values])

# analyse

sobol_indices = []
for idx in range(num**2):
    sobol_indices += [sobol.analyze(problem, y[:, idx,:].reshape(-1))]


S1s = np.array([sobol_indices[idx]['S1'] for idx in range(num**2)])




print(problem)

# problem.plot()
# plt.show()
#
problem.analyze_delta(num_resamples=5)
problem.plot()
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].bar([0, 1], sobol_indices[0]['S1'])
ax[0].set_xticks([0, 1], ['x1', 'x2'])
ax[0].set_title('x1_next')
ax[1].bar([0, 1], sobol_indices[1]['S1'])
ax[1].set_xticks([0, 1], ['x1', 'x2'])
ax[1].set_title('x2_next')
plt.show()
