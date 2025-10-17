"""
Random number generation with variance reduction techniques.
"""
import numpy as np
from scipy.stats import qmc, norm
from ..core.models import SimulationConfig


def generate_randoms(config: SimulationConfig) -> np.ndarray:
    """
    Generate random numbers based on simulation configuration.
    
    Returns:
    --------
    randoms : ndarray
        Shape (n_steps, n_paths_effective, 2) where n_paths_effective
        may be 2*n_paths for antithetic variates
    """
    if config.use_sobol:
        return _generate_sobol_randoms(config)
    elif config.use_antithetic:
        return _generate_antithetic_randoms(config)
    else:
        return _generate_standard_randoms(config)


def _generate_standard_randoms(config: SimulationConfig) -> np.ndarray:
    """Generate standard independent random numbers."""
    return np.random.standard_normal((config.n_steps, config.n_paths, 2))


def _generate_antithetic_randoms(config: SimulationConfig) -> np.ndarray:
    """
    Generate antithetic variates for variance reduction.
    Returns double the number of paths with perfect negative correlation.
    """
    # Ensure even number of paths
    n_base_paths = config.n_paths // 2
    if config.n_paths % 2 != 0:
        n_base_paths += 1
    
    # Generate base random numbers
    Z_base = np.random.standard_normal((config.n_steps, n_base_paths, 2))
    
    # Create antithetic pairs
    randoms = np.zeros((config.n_steps, n_base_paths * 2, 2))
    randoms[:, :n_base_paths, :] = Z_base
    randoms[:, n_base_paths:, :] = -Z_base
    
    return randoms


def _generate_sobol_randoms(config: SimulationConfig) -> np.ndarray:
    """
    Generate Sobol quasi-random numbers for improved convergence.
    Typically provides 5-10x better convergence than pseudo-random.
    """
    # Total dimensions needed: 2 per time step
    n_dims = 2 * config.n_steps
    
    # Generate Sobol sequence in [0,1]^n_dims
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=config.seed)
    sobol_uniform = sampler.random(n=config.n_paths)
    
    # Transform to standard normal
    Z_normals = norm.ppf(sobol_uniform)
    
    # Reshape to (n_steps, n_paths, 2)
    randoms = Z_normals.reshape((config.n_paths, config.n_steps, 2))
    randoms = np.transpose(randoms, (1, 0, 2))  # (n_steps, n_paths, 2)
    
    return randoms


def estimate_variance_reduction(payoffs_standard: np.ndarray, 
                              payoffs_variance_reduced: np.ndarray) -> dict:
    """
    Estimate the variance reduction achieved by a technique.
    
    Returns:
    --------
    dict with variance reduction statistics
    """
    var_standard = np.var(payoffs_standard, ddof=1)
    var_reduced = np.var(payoffs_variance_reduced, ddof=1)
    
    if var_standard == 0:
        variance_reduction = 0.0
    else:
        variance_reduction = (1 - var_reduced / var_standard) * 100
    
    return {
        'variance_standard': var_standard,
        'variance_reduced': var_reduced,
        'variance_reduction_pct': variance_reduction,
        'efficiency_ratio': var_standard / var_reduced if var_reduced > 0 else np.inf
    }