# Heston Monte Carlo Simulation

**Advanced Stochastic Volatility Modelling**

A comprehensive implementation of the Heston stochastic volatility model with Monte Carlo simulation, featuring multiple discretisation schemes, variance reduction techniques, and detailed financial analysis.

## Project Overview

This project provides a modern implementation of the Heston stochastic volatility model, featuring:

- **Multiple Discretisation Schemes**: Euler and Milstein for different accuracy requirements
- **Variance Reduction**: Antithetic variates and Sobol sequences for enhanced convergence
- **Comprehensive Analysis**: Option pricing, Greeks calculation, and model validation
- **Interactive Visualisation**: Detailed Jupyter notebook with financial insights

## Key Features

| Feature | Description |
|---------|-------------|
| **Heston Model** | Full stochastic volatility implementation with correlation |
| **Multiple Schemes** | Euler & Milstein discretisation |
| **Variance Reduction** | Antithetic variates & Sobol sequences |
| **Option Pricing** | European calls/puts with Greeks |
| **Path Simulation** | Vectorised Monte Carlo implementation |

## Heston Stochastic Volatility Model

The Heston model captures realistic equity dynamics through stochastic volatility:

```
dS/S = r dt + √V dW₁
dV = κ(θ - V)dt + η√V dW₂  
dW₁dW₂ = ρ dt
```

**Key Properties:**
- **Mean Reversion**: Volatility reverts to long-term level θ with speed κ
- **Volatility of Volatility**: Parameter η controls variance clustering
- **Leverage Effect**: Negative correlation ρ between returns and volatility
- **Closed-Form Solutions**: Available for European options via characteristic functions

## Quick Start

### Interactive Analysis
```bash
jupyter notebook Heston_Analysis.ipynb
```

### Direct Usage
```python
from heston_mc import HestonConfig, SimulationConfig, HestonPricer

# Configure model parameters
heston_config = HestonConfig(
    S0=100.0, K=100.0, r=0.05, T=1.0, V0=0.04,
    kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
)

# Configure simulation
sim_config = SimulationConfig(
    n_paths=50000, n_steps=100, scheme='milstein',
    use_antithetic=True, seed=42
)

# Price options
pricer = HestonPricer(heston_config)
call_result = pricer.price_call(sim_config)
print(f"Call price: {call_result.price:.4f}")
```


## Results & Validation

### Model Accuracy
- **Call Option Price**: $10.43 ± $0.08
- **Put Option Price**: $5.52 ± $0.06  
- **Put-Call Parity**: 0.68% error

### Variance Reduction
- **Antithetic Variates**: 30-50% variance reduction
- **Sobol Sequences**: Superior convergence vs pseudo-random

### Financial Insights
- **Volatility Clustering**: Visible in simulated paths
- **Leverage Effect**: Negative correlation drives realistic dynamics
- **Mean Reversion**: Long-term volatility stability

## Model Implementation

### Discretisation Schemes

#### Euler Scheme
```python
S_{t+1} = S_t exp((r - V_t/2)Δt + √V_t √Δt W₁)
V_{t+1} = max(0, V_t + κ(θ - V_t)Δt + η√V_t √Δt W₂)
```

#### Milstein Scheme (Higher Order)
```python
V_{t+1} = V_t + κ(θ - V_t)Δt + η√V_t √Δt W₂ + ¼η²Δt(W₂² - 1)
```

### Variance Reduction Techniques

- **Antithetic Variates**: Use paired random numbers (Z, -Z) to reduce variance
- **Sobol Sequences**: Quasi-random low-discrepancy sequences for better coverage
- **Control Variates**: Can be extended with correlated assets

## Parameter Calibration

**Standard Market Parameters:**
- `S₀ = $100`: Initial stock price
- `V₀ = 0.04`: Initial variance (20% volatility)  
- `κ = 2.0`: Mean reversion speed
- `θ = 0.04`: Long-term variance (20% volatility)
- `η = 0.3`: Volatility of volatility (30%)
- `ρ = -0.7`: Correlation (leverage effect)

**Feller Condition**: Ensures positive variance process
```
2κθ > η²  →  Prevents variance from reaching zero
```

## Installation & Requirements

```bash
pip install numpy scipy matplotlib jupyter
```

## Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── Heston_Analysis.ipynb        # Interactive financial analysis
└── heston_mc/                   # Modular implementation
    ├── core/                    # Core model components
    ├── pricing/                 # Option pricing tools
    ├── variance_reduction/      # Monte Carlo enhancements
    └── utils/                   # Utilities and validation
```

## Applications

- **Option Pricing**: European calls, puts, and exotic derivatives
- **Risk Management**: VaR calculations and stress testing
- **Portfolio Optimisation**: Multi-asset correlation modelling
- **Model Calibration**: Fitting to market volatility surfaces

## References

- **Heston, S. L. (1993)**: "A Closed-Form Solution for Options with Stochastic Volatility"
- **Gatheral, J. (2006)**: "The Volatility Surface: A Practitioner's Guide"
- **Glasserman, P. (2003)**: "Monte Carlo Methods in Financial Engineering"

---

**Author**: Yann Divet