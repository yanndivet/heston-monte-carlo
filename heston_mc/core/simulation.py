"""
Unified simulation engine for Heston Monte Carlo.
This module replaces the 6 duplicated simulation functions with a single, flexible engine.
"""
import numpy as np
from numba import jit
from typing import Optional, Union
from .models import HestonConfig, SimulationConfig
from .schemes import SCHEMES


@jit(nopython=True)
def _simulate_heston_core(S0: float, V0: float, r: float, kappa: float, theta: float,
                         eta: float, rho: float, dt: float, sqrt_dt: float,
                         n_paths: int, n_steps: int, randoms: np.ndarray,
                         scheme_func, return_paths: bool):
    """
    Unified JIT-compiled Heston simulation core.
    
    This single function replaces all 6 previous simulation functions,
    eliminating ~300 lines of code duplication.
    """
    # Initialize arrays
    S = np.full(n_paths, S0, dtype=np.float64)
    V = np.full(n_paths, V0, dtype=np.float64)
    
    # Conditional path storage (memory efficient)
    if return_paths:
        S_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        V_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        S_paths[:, 0] = S
        V_paths[:, 0] = V
    else:
        S_paths = np.empty((1, 1), dtype=np.float64)  # Minimal allocation
        V_paths = np.empty((1, 1), dtype=np.float64)
    
    # Pre-compute correlation terms
    sqrt_1_rho2 = np.sqrt(1 - rho**2)
    
    # Main simulation loop
    for step in range(n_steps):
        # Extract random numbers for this step
        Z1 = randoms[step, :, 0]
        Z2 = randoms[step, :, 1]
        
        # Correlated Brownian motions
        W_S = Z1
        W_V = rho * Z1 + sqrt_1_rho2 * Z2
        
        # Update using scheme-specific function
        S, V = scheme_func(S, V, W_S, W_V, r, kappa, theta, eta, dt, sqrt_dt)
        
        # Store paths if requested
        if return_paths:
            S_paths[:, step + 1] = S
            V_paths[:, step + 1] = V
    
    return S, S_paths, V_paths


def simulate_heston(heston_config: HestonConfig, 
                   sim_config: SimulationConfig,
                   randoms: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
    """
    Main simulation interface with automatic random number generation.
    
    Parameters:
    -----------
    heston_config : HestonConfig
        Heston model parameters
    sim_config : SimulationConfig  
        Simulation configuration
    randoms : ndarray, optional
        Pre-generated random numbers with shape (n_steps, n_paths, 2)
        If None, will generate based on sim_config settings
        
    Returns:
    --------
    If return_paths=False: final stock prices (n_paths,)
    If return_paths=True: (final_prices, S_paths, V_paths)
    """
    # Set random seed if specified
    if sim_config.seed is not None:
        np.random.seed(sim_config.seed)
    
    # Generate random numbers if not provided
    if randoms is None:
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from variance_reduction.random_generation import generate_randoms
        randoms = generate_randoms(sim_config)
    
    # Time discretization
    dt = heston_config.T / sim_config.n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Get scheme function
    if sim_config.scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme: {sim_config.scheme}")
    scheme_func = SCHEMES[sim_config.scheme]
    
    # Run simulation
    n_paths = randoms.shape[1]  # Actual number of paths (may include antithetic)
    
    S_final, S_paths, V_paths = _simulate_heston_core(
        heston_config.S0, heston_config.V0, heston_config.r,
        heston_config.kappa, heston_config.theta, heston_config.eta, heston_config.rho,
        dt, sqrt_dt, n_paths, sim_config.n_steps, randoms, scheme_func, sim_config.return_paths
    )
    
    if sim_config.return_paths:
        return S_final, S_paths, V_paths
    else:
        return S_final


class PathStorage:
    """Memory-efficient storage for simulation paths."""
    
    def __init__(self, n_paths: int, n_steps: int, store_variance: bool = True):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.S_paths = np.zeros((n_paths, n_steps + 1))
        if store_variance:
            self.V_paths = np.zeros((n_paths, n_steps + 1))
        else:
            self.V_paths = None
    
    def store_step(self, step: int, S: np.ndarray, V: Optional[np.ndarray] = None):
        """Store values for a specific time step."""
        self.S_paths[:, step] = S
        if self.V_paths is not None and V is not None:
            self.V_paths[:, step] = V
    
    def get_final_prices(self) -> np.ndarray:
        """Get final stock prices."""
        return self.S_paths[:, -1]