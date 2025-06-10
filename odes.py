# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# %%
t = sp.symbols('t')
mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3', cls=sp.Function)
mu1 = mu1(t); mu2 = mu2(t); mu3 = mu3(t)

x1, x2, lam = sp.symbols('x1 x2 lam')
z = sp.Matrix([x1, x2, lam])
mu = sp.Matrix([mu1, mu2, mu3])


# %%
# drift at mean
f1 = x1 - x1**3 + lam
f2 = -x2 + x1**2
f3 = 0
f = sp.Matrix([f1, f2, f3])


# %%
# jacobian evaluated at mean
J = f.jacobian(z)
J_mu = J.subs(dict(zip(z, mu)))

# %%
sigma = sp.symbols('sigma', positive=True)
sigma_sq = sigma**2 # variance of noise
Q = sp.diag(sigma_sq, sigma_sq, 0) # only noise in x1 and x2
mu_dot = f.subs(dict(zip(z, mu)))

Sigma = sp.MatrixSymbol('Sigma', 3, 3)  # covariance matrix
Sigma_dot = J_mu * Sigma + Sigma * J_mu.T + Q

# %%
mean_odes = [
    sp.Eq(sp.diff(mu1, t), mu_dot[0]),
    sp.Eq(sp.diff(mu2, t), mu_dot[1]),
    sp.Eq(sp.diff(mu3, t), mu_dot[2])
]

cov_odes = [
    sp.Eq(sp.Derivative(Sigma[i, j], t), sp.simplify(Sigma_dot[i, j]))
    for i in range(3) for j in range(3)
]

from IPython.display import display, Math
for ode in mean_odes + cov_odes:
    display(Math(sp.latex(ode)))

# %%

# params
T = 10.0
dt = 0.01
steps = int(T / dt)
num_neurons = 1000
sigma = 0.2
lambda_val = 0.5

# %%
X = np.zeros((steps, num_neurons, 2)) # [time, neurons, (x1, x2)]
X[0, :, :] = np.random.randn(num_neurons, 2) * 0.1 # small init perturbation

# drift function
def f(x1, x2, lam):
    dx1 = x1 - x1**3 + lam
    dx2 = -x2 + x1**2
    return np.stack([dx1, dx2], axis=-1)

# %%
# simulate ensemble
for n in range(steps - 1):
    x1 = X[n, :, 0]
    x2 = X[n, :, 1]
    drift = f(x1, x2, lambda_val)
    noise = sigma * np.sqrt(dt) * np.random.randn(num_neurons, 2)
    X[n+1, :, :] = X[n, :, :] + drift * dt + noise

# %%
# compute empirical mean and cov
mu_emp = np.mean(X, axis=1)
cov_emp = np.array([np.cov(X[n, :, :].T) for n in range(steps)])

plt.plot(np.linspace(0, T, steps), mu_emp[:, 0], label='Empirical Mu_1(t)')
plt.plot(np.linspace(0, T, steps), mu_emp[:, 1], label='Empirical Mu_2(t)')
plt.xlabel('Time')
plt.ylabel('Empirical Mean')
plt.title('Empirical Mean of Neuron Dynamics')
plt.legend()
plt.show()

# %%



