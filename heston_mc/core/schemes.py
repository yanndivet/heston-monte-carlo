"""
Discretization schemes for the Heston stochastic volatility model.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def euler_step(S: np.ndarray, V: np.ndarray, W_S: np.ndarray, W_V: np.ndarray,
               r: float, kappa: float, theta: float, eta: float, 
               dt: float, sqrt_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Single Euler discretization step for Heston model.
    
    Parameters:
    -----------
    S, V : ndarray
        Current stock price and variance arrays
    W_S, W_V : ndarray  
        Brownian increments for stock and variance
    r, kappa, theta, eta : float
        Model parameters
    dt, sqrt_dt : float
        Time step and its square root
        
    Returns:
    --------
    S_new, V_new : ndarray
        Updated stock price and variance
    """
    sqrt_V = np.sqrt(np.maximum(V, 0))
    
    # Stock price update (exact for log-normal part)
    S_new = S * np.exp((r - 0.5 * V) * dt + sqrt_V * sqrt_dt * W_S)
    
    # Variance update (Euler scheme)
    V_new = np.maximum(0, V + kappa * (theta - V) * dt + eta * sqrt_V * sqrt_dt * W_V)
    
    return S_new, V_new


@jit(nopython=True) 
def milstein_step(S: np.ndarray, V: np.ndarray, W_S: np.ndarray, W_V: np.ndarray,
                  r: float, kappa: float, theta: float, eta: float,
                  dt: float, sqrt_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Single Milstein discretization step for Heston model.
    Adds second-order correction term for better accuracy.
    """
    sqrt_V = np.sqrt(np.maximum(V, 0))
    
    # Stock price update (same as Euler)
    S_new = S * np.exp((r - 0.5 * V) * dt + sqrt_V * sqrt_dt * W_S)
    
    # Variance update with Milstein correction
    V_new = V + kappa * (theta - V) * dt + eta * sqrt_V * sqrt_dt * W_V + \
            0.25 * eta**2 * dt * (W_V**2 - 1)
    V_new = np.maximum(0, V_new)
    
    return S_new, V_new


# Scheme lookup for dynamic dispatch
SCHEMES = {
    'euler': euler_step,
    'milstein': milstein_step
}