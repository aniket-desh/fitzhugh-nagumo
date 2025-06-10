# FitzHugh-Nagumo Neural Network Analysis

This project implements and analyzes the FitzHugh-Nagumo neural network model with stochastic dynamics, focusing on moment closure analysis and ensemble simulations.

## Overview

The FitzHugh-Nagumo model is a simplified version of the Hodgkin-Huxley model that describes the electrical activity of neurons. This implementation includes:

- Stochastic dynamics with noise in the state variables
- Moment closure analysis for mean and covariance evolution
- Ensemble simulations of multiple neurons
- Empirical analysis of mean and covariance dynamics

## Mathematical Model

The system is described by the following stochastic differential equations:

$$\begin{align*}
\frac{dx_1}{dt} &= x_1 - x_1^3 + \lambda + \sigma \xi_1,\quad \frac{dx_2}{dt} = -x_2 + x_1^2 + \sigma \xi_2,\quad \frac{d\lambda}{dt} = 0
\end{align*}$$

where:
- $x_1, x_2$ are the state variables
- $\lambda$ is a constant parameter
- $\sigma$ is the noise intensity
- $\xi_1, \xi_2$ are independent white noise processes

The moment equations for the mean $\mu$ and covariance $\Sigma$ are derived using the Jacobian matrix $J$ evaluated at the mean:

$$\begin{align*}
\frac{d\mu}{dt} &= f(\mu),\quad \frac{d\Sigma}{dt} = J(\mu)\Sigma + \Sigma J(\mu)^T + Q
\end{align*}$$

where $Q$ is the noise covariance matrix.

## Implementation Details

The project uses:
- `sympy` for symbolic computation of moment equations
- `numpy` for numerical simulations
- `matplotlib` and `seaborn` for visualization
- `scikit-learn` for error metrics

### Key Components

1. **Symbolic Analysis** (`odes.py`):
   - Derives the moment equations using symbolic computation
   - Computes the Jacobian and covariance evolution equations
   - Generates the system of ODEs for mean and covariance

2. **Numerical Simulation**:
   - Implements ensemble simulation of multiple neurons
   - Computes empirical statistics from the ensemble
   - Visualizes the dynamics of mean and covariance

## Future Work

- Implement moment closure approximations
- Add bifurcation analysis
- Include more sophisticated noise models
- Add parameter estimation capabilities 