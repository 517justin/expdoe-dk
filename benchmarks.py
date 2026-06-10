"""
benchmarks.py
=============
Mathematical benchmark functions for DOE → BO experiments.
All functions accept X ∈ [0,1]^d (unit hypercube) and return scalar values.
"""

import numpy as np


def branin_2d(X: np.ndarray) -> np.ndarray:
    """
    Branin function (2D). Global optimum ≈ 0.397887 (three optima).
    Input: X ∈ [0,1]^2, mapped to x1 ∈ [-5,10], x2 ∈ [0,15]
    """
    X = np.atleast_2d(X)
    x1 = X[:, 0] * 15.0 - 5.0
    x2 = X[:, 1] * 15.0
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def hartmann_3d(X: np.ndarray) -> np.ndarray:
    """
    Hartmann 3D function. Global optimum ≈ -3.8628 at (0.114, 0.556, 0.852).
    Input: X ∈ [0,1]^3
    """
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [ 381, 5743, 8828],
    ])
    result = np.zeros(X.shape[0])
    for i in range(4):
        inner = np.sum(A[i] * (X - P[i])**2, axis=1)
        result -= alpha[i] * np.exp(-inner)
    return result


def rosenbrock_nd(X: np.ndarray) -> np.ndarray:
    """
    Rosenbrock function (nD). Global optimum = 0 at (1,...,1).
    Input: X ∈ [0,1]^d, mapped to [-2,2]^d
    """
    X = np.atleast_2d(X)
    X_scaled = X * 4.0 - 2.0
    result = np.zeros(X_scaled.shape[0])
    for i in range(X_scaled.shape[1] - 1):
        xi = X_scaled[:, i]
        xi1 = X_scaled[:, i + 1]
        result += 100.0 * (xi1 - xi**2)**2 + (1.0 - xi)**2
    return result


def ackley_nd(X: np.ndarray, a: float = 20.0, b: float = 0.2,
              c: float = 2 * np.pi) -> np.ndarray:
    """
    Ackley function (nD). Global optimum = 0 at origin (mapped: x=0.5).
    Input: X ∈ [0,1]^d, mapped to [-32.768, 32.768]^d
    """
    X = np.atleast_2d(X)
    X_scaled = X * 65.536 - 32.768
    term1 = -a * np.exp(-b * np.sqrt(np.mean(X_scaled**2, axis=1)))
    term2 = -np.exp(np.mean(np.cos(c * X_scaled), axis=1))
    return term1 + term2 + a + np.e
