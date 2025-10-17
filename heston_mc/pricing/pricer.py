"""
Main pricing interface for Heston Monte Carlo.
"""
import numpy as np
from dataclasses import replace
from typing import Optional

from ..core.models import HestonConfig, SimulationConfig, PricingResult, GreeksResult
from ..core.simulation import simulate_heston


class HestonPricer:
    """
    Main interface for Heston option pricing with Monte Carlo simulation.
    
    This class replaces the monolithic HestonMonteCarlo class with a cleaner,
    more modular design using immutable configuration objects.
    """
    
    def __init__(self, heston_config: HestonConfig):
        """
        Initialize pricer with Heston model configuration.
        
        Parameters:
        -----------
        heston_config : HestonConfig
            Immutable configuration for Heston model parameters
        """
        self.config = heston_config
    
    def price_call(self, sim_config: SimulationConfig) -> PricingResult:
        """
        Price a European call option.
        
        Parameters:
        -----------
        sim_config : SimulationConfig
            Simulation parameters
            
        Returns:
        --------
        PricingResult with price statistics
        """
        # Simulate final stock prices
        S_final = simulate_heston(self.config, sim_config)
        
        # Calculate call payoffs
        payoffs = np.maximum(S_final - self.config.K, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-self.config.r * self.config.T) * payoffs
        
        return PricingResult.from_payoffs(discounted_payoffs, sim_config)
    
    def price_put(self, sim_config: SimulationConfig) -> PricingResult:
        """
        Price a European put option.
        """
        S_final = simulate_heston(self.config, sim_config)
        payoffs = np.maximum(self.config.K - S_final, 0)
        discounted_payoffs = np.exp(-self.config.r * self.config.T) * payoffs
        
        return PricingResult.from_payoffs(discounted_payoffs, sim_config)
    
    def calculate_greeks(self, sim_config: SimulationConfig, 
                        bump_pct: float = 0.01) -> GreeksResult:
        """
        Calculate option Greeks using finite differences.
        
        This implementation uses immutable config objects to avoid
        the state mutation problems in the original code.
        
        Parameters:
        -----------
        sim_config : SimulationConfig
            Simulation configuration
        bump_pct : float
            Percentage bump for finite differences
            
        Returns:
        --------
        GreeksResult with all Greeks
        """
        # Ensure reproducible results for Greeks
        sim_config_fixed_seed = replace(sim_config, seed=42)
        
        # Base price
        base_price = self.price_call(sim_config_fixed_seed).price
        
        # Delta and Gamma: bump stock price
        dS = self.config.S0 * bump_pct
        
        config_S_up = replace(self.config, S0=self.config.S0 + dS)
        config_S_down = replace(self.config, S0=self.config.S0 - dS)
        
        pricer_S_up = HestonPricer(config_S_up)
        pricer_S_down = HestonPricer(config_S_down)
        
        price_S_up = pricer_S_up.price_call(sim_config_fixed_seed).price
        price_S_down = pricer_S_down.price_call(sim_config_fixed_seed).price
        
        delta = (price_S_up - price_S_down) / (2 * dS)
        gamma = (price_S_up - 2 * base_price + price_S_down) / (dS ** 2)
        
        # Vega: bump initial volatility
        sigma0 = np.sqrt(self.config.V0)
        d_sigma = sigma0 * bump_pct
        V0_up = (sigma0 + d_sigma) ** 2
        
        config_V_up = replace(self.config, V0=V0_up)
        pricer_V_up = HestonPricer(config_V_up)
        price_V_up = pricer_V_up.price_call(sim_config_fixed_seed).price
        
        vega = (price_V_up - base_price) / d_sigma
        
        # Theta: bump time to maturity
        dT = 1 / 365  # One day
        config_T_down = replace(self.config, T=self.config.T - dT)
        pricer_T_down = HestonPricer(config_T_down)
        price_T_down = pricer_T_down.price_call(sim_config_fixed_seed).price
        
        theta = (price_T_down - base_price) / dT
        
        # Rho: bump risk-free rate
        dr = 0.01 * bump_pct  # 0.01% rate change
        config_r_up = replace(self.config, r=self.config.r + dr)
        pricer_r_up = HestonPricer(config_r_up)
        price_r_up = pricer_r_up.price_call(sim_config_fixed_seed).price
        
        rho = (price_r_up - base_price) / dr
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            base_price=base_price
        )
    
    def simulate_paths(self, sim_config: SimulationConfig):
        """
        Simulate price and variance paths.
        
        Returns:
        --------
        S_final, S_paths, V_paths if return_paths=True
        S_final if return_paths=False
        """
        return simulate_heston(self.config, sim_config)
    
    def verify_put_call_parity(self, sim_config: SimulationConfig) -> dict:
        """
        Verify put-call parity: C - P = S0 - K*exp(-r*T)
        
        Returns:
        --------
        dict with parity check results
        """
        call_result = self.price_call(sim_config)
        put_result = self.price_put(sim_config)
        
        pcp_lhs = call_result.price - put_result.price
        pcp_rhs = self.config.S0 - self.config.K * np.exp(-self.config.r * self.config.T)
        difference = abs(pcp_lhs - pcp_rhs)
        
        return {
            'call_price': call_result.price,
            'put_price': put_result.price,
            'C_minus_P': pcp_lhs,
            'S_minus_K_discounted': pcp_rhs,
            'difference': difference,
            'relative_error': difference / abs(pcp_rhs) if pcp_rhs != 0 else np.inf
        }