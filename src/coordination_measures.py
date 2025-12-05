"""
Coordination Measures Module

Core functions for computing temporal coordination capacity (Φ_d, Φ_f, Φ_c)
using information-theoretic measures.

Theory:
- Duration (Φ_d): Past → Present constraint (normalized mutual information)
- Frequency (Φ_f): Present → Future projection (normalized transfer entropy)
- Coordination (Φ_c): Balanced temporal integration (geometric mean)

References:
    Cabot, Z. (2025). Hemispheric Dialogue Across Species.
    Proceedings of the Royal Society B.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from typing import Tuple, Optional


def compute_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """
    Compute mutual information I(X; Y) using k-nearest neighbor estimation.
    
    Uses the Kraskov-Stögbauer-Grassberger (KSG) estimator.
    
    Parameters
    ----------
    x : np.ndarray
        First variable, shape (n_samples,) or (n_samples, n_features)
    y : np.ndarray
        Second variable, shape (n_samples,) or (n_samples, n_features)
    k : int
        Number of nearest neighbors (default: 3)
    
    Returns
    -------
    float
        Estimated mutual information in nats
        
    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). 
    Estimating mutual information. Physical Review E, 69(6), 066138.
    """
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    
    n = len(x)
    
    # Joint space
    xy = np.hstack([x, y])
    
    # Build trees
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    
    # Find k-th neighbor distances in joint space
    distances, _ = tree_xy.query(xy, k=k+1)
    epsilon = distances[:, -1]
    
    # Count neighbors within epsilon in marginal spaces
    n_x = np.array([len(tree_x.query_ball_point(xi, eps)) - 1 
                    for xi, eps in zip(x, epsilon)])
    n_y = np.array([len(tree_y.query_ball_point(yi, eps)) - 1 
                    for yi, eps in zip(y, epsilon)])
    
    # KSG estimator
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    
    return max(0, mi)  # MI is non-negative


def compute_entropy(x: np.ndarray, k: int = 3) -> float:
    """
    Compute entropy H(X) using k-nearest neighbor estimation.
    
    Parameters
    ----------
    x : np.ndarray
        Variable, shape (n_samples,) or (n_samples, n_features)
    k : int
        Number of nearest neighbors (default: 3)
    
    Returns
    -------
    float
        Estimated entropy in nats
    """
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    n, d = x.shape
    
    tree = cKDTree(x)
    distances, _ = tree.query(x, k=k+1)
    epsilon = distances[:, -1]
    
    # Volume of unit ball in d dimensions
    cd = np.pi**(d/2) / np.exp(np.lgamma(d/2 + 1))
    
    # Kozachenko-Leonenko estimator
    h = digamma(n) - digamma(k) + d * np.mean(np.log(2 * epsilon + 1e-10)) + np.log(cd)
    
    return h


def compute_transfer_entropy(
    source: np.ndarray, 
    target: np.ndarray, 
    tau: int = 1, 
    k: int = 3,
    condition: Optional[np.ndarray] = None
) -> float:
    """
    Compute transfer entropy TE(X → Y | Z) using k-NN estimation.
    
    Transfer entropy measures information flow from source to target,
    conditioned on the target's past (and optionally additional variables).
    
    TE(X → Y) = I(X_t; Y_{t+τ} | Y_t)
    
    Parameters
    ----------
    source : np.ndarray
        Source time series, shape (n_samples,)
    target : np.ndarray
        Target time series, shape (n_samples,)
    tau : int
        Time lag (default: 1)
    k : int
        Number of nearest neighbors (default: 3)
    condition : np.ndarray, optional
        Additional conditioning variable
    
    Returns
    -------
    float
        Estimated transfer entropy in nats
    """
    n = len(source) - tau
    
    # Construct variables
    x_t = source[:-tau][:n]         # Source at time t
    y_t = target[:-tau][:n]         # Target at time t (past)
    y_future = target[tau:][:n]     # Target at time t+τ (future)
    
    # Reshape for k-NN
    x_t = x_t.reshape(-1, 1)
    y_t = y_t.reshape(-1, 1)
    y_future = y_future.reshape(-1, 1)
    
    if condition is not None:
        cond = condition[:-tau][:n].reshape(-1, 1)
        y_t = np.hstack([y_t, cond])
    
    # TE = I(X_t; Y_future | Y_past)
    # = H(Y_future | Y_past) - H(Y_future | X_t, Y_past)
    # = I(X_t, Y_past; Y_future) - I(Y_past; Y_future)
    
    # Compute as: I(X_t; Y_future, Y_past) - I(X_t; Y_past)
    xy_past = np.hstack([x_t, y_t])
    y_both = np.hstack([y_future, y_t])
    
    mi_full = compute_mutual_information(x_t, y_both, k=k)
    mi_past = compute_mutual_information(x_t, y_t, k=k)
    
    te = mi_full - mi_past
    
    return max(0, te)


def compute_phi_d(signal: np.ndarray, tau: int = 1, k: int = 3) -> float:
    """
    Compute Duration (Φ_d): normalized mutual information between past and present.
    
    Φ_d = I(X_{t-τ}; X_t) / H(X_t)
    
    Measures how much the past constrains the present (temporal inertia).
    
    Parameters
    ----------
    signal : np.ndarray
        Time series, shape (n_samples,)
    tau : int
        Time lag (default: 1)
    k : int
        Number of nearest neighbors (default: 3)
    
    Returns
    -------
    float
        Duration value in [0, 1]
    """
    n = len(signal) - tau
    
    x_past = signal[:-tau][:n].reshape(-1, 1)
    x_present = signal[tau:][:n].reshape(-1, 1)
    
    mi = compute_mutual_information(x_past, x_present, k=k)
    h = compute_entropy(x_present, k=k)
    
    if h <= 0:
        return 0.0
    
    phi_d = mi / h
    return np.clip(phi_d, 0, 1)


def compute_phi_f(
    signal: np.ndarray, 
    tau: int = 1, 
    k: int = 3
) -> float:
    """
    Compute Frequency (Φ_f): normalized conditional transfer entropy.
    
    Φ_f = TE(X_t → X_{t+τ} | X_{t-τ}) / H(X_{t+τ})
    
    Measures projection toward future states beyond what past already predicts.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series, shape (n_samples,)
    tau : int
        Time lag (default: 1)
    k : int
        Number of nearest neighbors (default: 3)
    
    Returns
    -------
    float
        Frequency value in [0, 1]
    """
    n = len(signal) - 2 * tau
    
    x_past = signal[:-2*tau][:n]
    x_present = signal[tau:-tau][:n]
    x_future = signal[2*tau:][:n]
    
    # Conditional transfer entropy
    te = compute_transfer_entropy(x_present, x_future, tau=1, k=k, 
                                   condition=x_past)
    
    h_future = compute_entropy(x_future.reshape(-1, 1), k=k)
    
    if h_future <= 0:
        return 0.0
    
    phi_f = te / h_future
    return np.clip(phi_f, 0, 1)


def compute_phi_c(phi_d: float, phi_f: float) -> float:
    """
    Compute Coordination Capacity (Φ_c) from Duration and Frequency.
    
    Φ_c = 4 × √[Φ_d(1-Φ_d) × Φ_f(1-Φ_f)]
    
    Maximum coordination (Φ_c = 1) when Φ_d = Φ_f = 0.5 (balanced integration).
    Minimum (Φ_c = 0) when either component is at extremes (0 or 1).
    
    Parameters
    ----------
    phi_d : float
        Duration value in [0, 1]
    phi_f : float
        Frequency value in [0, 1]
    
    Returns
    -------
    float
        Coordination capacity in [0, 1]
    """
    # Ensure inputs are in valid range
    phi_d = np.clip(phi_d, 0, 1)
    phi_f = np.clip(phi_f, 0, 1)
    
    # Geometric mean of variance-like terms
    term_d = phi_d * (1 - phi_d)
    term_f = phi_f * (1 - phi_f)
    
    phi_c = 4 * np.sqrt(term_d * term_f)
    
    return phi_c


def compute_coordination_from_signal(
    signal: np.ndarray, 
    tau: int = 1, 
    k: int = 3
) -> Tuple[float, float, float]:
    """
    Compute all three coordination measures from a time series.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series, shape (n_samples,)
    tau : int
        Time lag (default: 1)
    k : int
        Number of nearest neighbors (default: 3)
    
    Returns
    -------
    tuple
        (phi_d, phi_f, phi_c) - Duration, Frequency, Coordination Capacity
    """
    phi_d = compute_phi_d(signal, tau=tau, k=k)
    phi_f = compute_phi_f(signal, tau=tau, k=k)
    phi_c = compute_phi_c(phi_d, phi_f)
    
    return phi_d, phi_f, phi_c


# Example usage
if __name__ == '__main__':
    # Generate sample signal
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(len(t))
    
    phi_d, phi_f, phi_c = compute_coordination_from_signal(signal)
    
    print(f"Duration (Φ_d): {phi_d:.3f}")
    print(f"Frequency (Φ_f): {phi_f:.3f}")
    print(f"Coordination (Φ_c): {phi_c:.3f}")
