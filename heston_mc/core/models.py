"""
Core data models and configuration for Heston Monte Carlo simulation.
"""
import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HestonConfig:
    """
    Configuration for Heston stochastic volatility model.
    
    Model equations:
        dS/S = r dt + sqrt(V) dW^S
        dV = kappa(theta - V)dt + eta*sqrt(V) dW^V
        dW^V dW^S = rho dt
    """
    S0: float = 100.0       # Initial stock price
    K: float = 100.0        # Strike price  
    r: float = 0.05         # Risk-free rate
    T: float = 1.0          # Time to maturity
    V0: float = 0.04        # Initial variance
    kappa: float = 2.0      # Mean reversion speed
    theta: float = 0.04     # Long-term variance level
    eta: float = 0.3        # Volatility of volatility
    rho: float = -0.7       # Correlation between stock and variance Brownian motions
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()
        self._check_feller_condition()
    
    def _validate_parameters(self):
        """Validate parameter bounds and constraints."""
        if self.S0 <= 0:
            raise ValueError(f"Initial stock price must be positive: S0={self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive: K={self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive: T={self.T}")
        if self.V0 < 0:
            raise ValueError(f"Initial variance must be non-negative: V0={self.V0}")
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive: kappa={self.kappa}")
        if self.theta < 0:
            raise ValueError(f"Long-term variance must be non-negative: theta={self.theta}")
        if self.eta < 0:
            raise ValueError(f"Volatility of volatility must be non-negative: eta={self.eta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation must be in [-1, 1]: rho={self.rho}")
    
    def _check_feller_condition(self):
        """
        Check the Feller condition: 2*kappa*theta > eta^2
        If violated, variance can reach zero with positive probability.
        """
        feller_lhs = 2 * self.kappa * self.theta
        feller_rhs = self.eta ** 2
        
        if feller_lhs <= feller_rhs:
            warnings.warn(
                f"Feller condition violated: 2κθ = {feller_lhs:.6f} <= η² = {feller_rhs:.6f}. "
                f"The variance process can reach zero with positive probability, "
                f"which may cause numerical issues.",
                UserWarning
            )


@dataclass 
class SimulationConfig:
    """Configuration for Monte Carlo simulation parameters."""
    n_paths: int = 10000
    n_steps: int = 100
    scheme: str = 'euler'           # 'euler' or 'milstein'
    use_antithetic: bool = False
    use_sobol: bool = False
    return_paths: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate simulation parameters."""
        if self.n_paths <= 0:
            raise ValueError(f"Number of paths must be positive: n_paths={self.n_paths}")
        if self.n_steps <= 0:
            raise ValueError(f"Number of steps must be positive: n_steps={self.n_steps}")
        if self.scheme not in ['euler', 'milstein']:
            raise ValueError(f"Unknown scheme: {self.scheme}. Use 'euler' or 'milstein'")
        if self.use_sobol and self.use_antithetic:
            raise ValueError("Cannot use both Sobol and antithetic variates simultaneously")


@dataclass
class PricingResult:
    """Result container for option pricing."""
    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    ci_width: float
    n_paths: int
    scheme: str
    use_antithetic: bool = False
    use_sobol: bool = False
    
    @classmethod
    def from_payoffs(cls, payoffs: np.ndarray, config: SimulationConfig) -> 'PricingResult':
        """Create PricingResult from array of discounted payoffs."""
        price = np.mean(payoffs)
        std_error = np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))
        ci_95 = 1.96 * std_error
        
        return cls(
            price=price,
            std_error=std_error,
            confidence_interval=(price - ci_95, price + ci_95),
            ci_width=ci_95,
            n_paths=config.n_paths,
            scheme=config.scheme,
            use_antithetic=config.use_antithetic,
            use_sobol=config.use_sobol
        )


@dataclass
class GreeksResult:
    """Result container for option Greeks."""
    delta: float
    gamma: float  
    vega: float
    theta: float
    rho: float
    base_price: float