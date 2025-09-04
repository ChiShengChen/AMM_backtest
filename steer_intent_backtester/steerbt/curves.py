"""
Liquidity distribution curves for CLMM strategies.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CurveBase:
    """Base class for liquidity distribution curves."""
    
    def __init__(self, **kwargs):
        self.max_bins = kwargs.get("max_bins", 10)
        self.mirror = kwargs.get("mirror", False)
        self.invert = kwargs.get("invert", False)
        
        # Add liquidity scaling factor to convert from USD to reasonable Uniswap V3 units
        # This prevents liquidity values from being too large and causing overflow
        self.liquidity_scale = kwargs.get("liquidity_scale", 0.001)  # Scale factor for liquidity
    
    def generate_distribution(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Total liquidity to distribute (in USD)
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        # Scale the liquidity to reasonable Uniswap V3 units
        # This prevents the liquidity values from being too large
        scaled_liquidity = total_liquidity * self.liquidity_scale
        
        # Ensure minimum liquidity to prevent zero values
        scaled_liquidity = max(scaled_liquidity, 100.0)
        
        return self._generate_distribution_impl(center_price, width_pct, scaled_liquidity)
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Implementation of liquidity distribution generation.
        This method should be overridden by subclasses.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        raise NotImplementedError("Subclasses must implement _generate_distribution_impl")
    
    def _apply_transforms(
        self,
        prices: List[float],
        liquidities: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Apply mirror and invert transformations."""
        if self.mirror:
            # Mirror around center
            center_idx = len(prices) // 2
            prices = prices[:center_idx + 1] + prices[center_idx:][::-1]
            liquidities = liquidities[:center_idx + 1] + liquidities[center_idx:][::-1]
        
        if self.invert:
            # Invert liquidity distribution
            max_liq = max(liquidities) if liquidities else 0
            liquidities = [max_liq - liq for liq in liquidities]
        
        return prices, liquidities


class LinearCurve(CurveBase):
    """Linear liquidity distribution curve."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.slope = kwargs.get("slope", 1.0)  # Slope of liquidity distribution
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate linear liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))  # Adaptive bin count
        bin_count = max(2, bin_count)  # Minimum 2 bins
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Generate linear liquidity distribution
        liquidities = []
        for i in range(bin_count):
            # Linear decrease from center
            distance_from_center = abs(i - bin_count/2) / (bin_count/2)
            liquidity = total_liquidity / bin_count * (1 - distance_from_center * self.slope)
            liquidities.append(max(liquidity, total_liquidity / bin_count * 0.1))  # Minimum liquidity
        
        # Apply transformations
        prices, liquidities = self._apply_transforms(prices, liquidities)
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            liquidity = liquidities[i]
            bins.append((lower, upper, liquidity))
        
        return bins


class GaussianCurve(CurveBase):
    """Gaussian (normal) liquidity distribution curve."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.std_dev = kwargs.get("std_dev", 0.5)  # Standard deviation multiplier
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate Gaussian liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))
        bin_count = max(3, bin_count)  # Minimum 3 bins for Gaussian
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Generate Gaussian liquidity distribution
        liquidities = []
        for i in range(bin_count):
            # Gaussian distribution centered at middle
            center_idx = bin_count / 2
            distance = (i - center_idx) / (bin_count / 2)
            gaussian = np.exp(-0.5 * (distance / self.std_dev) ** 2)
            liquidity = total_liquidity / bin_count * gaussian
            liquidities.append(max(liquidity, total_liquidity / bin_count * 0.05))
        
        # Apply transformations
        prices, liquidities = self._apply_transforms(prices, liquidities)
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            liquidity = liquidities[i]
            bins.append((lower, upper, liquidity))
        
        return bins


class SigmoidCurve(CurveBase):
    """Sigmoid liquidity distribution curve."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = kwargs.get("k", 5.0)  # Sigmoid steepness parameter
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate sigmoid liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))
        bin_count = max(3, bin_count)
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Generate sigmoid liquidity distribution
        liquidities = []
        for i in range(bin_count):
            # Sigmoid function
            x = (i - bin_count/2) / (bin_count/2) * 2  # Scale to [-2, 2]
            sigmoid = 1 / (1 + np.exp(-self.k * x))
            liquidity = total_liquidity / bin_count * sigmoid
            liquidities.append(max(liquidity, total_liquidity / bin_count * 0.1))
        
        # Apply transformations
        prices, liquidities = self._apply_transforms(prices, liquidities)
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            liquidity = liquidities[i]
            bins.append((lower, upper, liquidity))
        
        return bins


class LogarithmicCurve(CurveBase):
    """Logarithmic liquidity distribution curve."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base = kwargs.get("base", 2.0)  # Logarithm base
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate logarithmic liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))
        bin_count = max(3, bin_count)
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Generate logarithmic liquidity distribution
        liquidities = []
        for i in range(bin_count):
            # Logarithmic distribution
            log_factor = np.log(1 + i + 1) / np.log(self.base)
            liquidity = total_liquidity / bin_count * log_factor
            liquidities.append(max(liquidity, total_liquidity / bin_count * 0.1))
        
        # Apply transformations
        prices, liquidities = self._apply_transforms(prices, liquidities)
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            liquidity = liquidities[i]
            bins.append((lower, upper, liquidity))
        
        return bins


class BidAskCurve(CurveBase):
    """Bid-ask twin peaks liquidity distribution curve."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peak_separation = kwargs.get("peak_separation", 0.1)  # Separation between peaks
        self.peak_width = kwargs.get("peak_width", 0.05)  # Width of each peak
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate bid-ask twin peaks liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))
        bin_count = max(5, bin_count)  # Minimum 5 bins for twin peaks
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Generate twin peaks liquidity distribution
        liquidities = []
        for i in range(bin_count):
            price = prices[i]
            
            # Calculate distance from each peak
            peak1 = center_price - self.peak_separation * center_price / 2
            peak2 = center_price + self.peak_separation * center_price / 2
            
            dist1 = abs(price - peak1) / (self.peak_width * center_price)
            dist2 = abs(price - peak2) / (self.peak_width * center_price)
            
            # Gaussian peaks
            gaussian1 = np.exp(-0.5 * dist1 ** 2)
            gaussian2 = np.exp(-0.5 * dist2 ** 2)
            
            # Combine peaks
            combined = (gaussian1 + gaussian2) / 2
            liquidity = total_liquidity / bin_count * combined
            liquidities.append(max(liquidity, total_liquidity / bin_count * 0.05))
        
        # Apply transformations
        prices, liquidities = self._apply_transforms(prices, liquidities)
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            liquidity = liquidities[i]
            bins.append((lower, upper, liquidity))
        
        return bins


class UniformCurve(CurveBase):
    """Uniform (equal) liquidity distribution curve."""
    
    def _generate_distribution_impl(
        self,
        center_price: float,
        width_pct: float,
        total_liquidity: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate uniform liquidity distribution.
        
        Args:
            center_price: Center price for distribution
            width_pct: Width as percentage of center price
            total_liquidity: Scaled liquidity value
            
        Returns:
            List of (lower_price, upper_price, liquidity) tuples
        """
        width = center_price * width_pct / 100
        
        # Generate price bins
        bin_count = min(self.max_bins, int(width_pct / 2))
        bin_count = max(2, bin_count)
        
        prices = np.linspace(center_price - width/2, center_price + width/2, bin_count + 1)
        
        # Equal liquidity distribution
        liquidity_per_bin = total_liquidity / bin_count
        
        # Create bins
        bins = []
        for i in range(len(prices) - 1):
            lower = prices[i]
            upper = prices[i + 1]
            bins.append((lower, upper, liquidity_per_bin))
        
        return bins


class CurveFactory:
    """Factory for creating liquidity distribution curves."""
    
    _curves = {
        "linear": LinearCurve,
        "gaussian": GaussianCurve,
        "sigmoid": SigmoidCurve,
        "logarithmic": LogarithmicCurve,
        "bid_ask": BidAskCurve,
        "uniform": UniformCurve,
    }
    
    @classmethod
    def create_curve(cls, curve_type: str, **kwargs) -> CurveBase:
        """
        Create a liquidity distribution curve.
        
        Args:
            curve_type: Type of curve to create
            **kwargs: Curve-specific parameters
            
        Returns:
            Curve instance
        """
        if curve_type not in cls._curves:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        return cls._curves[curve_type](**kwargs)
    
    @classmethod
    def get_available_curves(cls) -> List[str]:
        """Get list of available curve types."""
        return list(cls._curves.keys())
    
    @classmethod
    def get_curve_params(cls, curve_type: str) -> Dict[str, Any]:
        """Get default parameters for a curve type."""
        if curve_type == "linear":
            return {"slope": 1.0, "max_bins": 10}
        elif curve_type == "gaussian":
            return {"std_dev": 0.5, "max_bins": 10}
        elif curve_type == "sigmoid":
            return {"k": 5.0, "max_bins": 10}
        elif curve_type == "logarithmic":
            return {"base": 2.0, "max_bins": 10}
        elif curve_type == "bid_ask":
            return {"peak_separation": 0.1, "peak_width": 0.05, "max_bins": 10}
        elif curve_type == "uniform":
            return {"max_bins": 10}
        else:
            return {}
