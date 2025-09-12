"""
Fluid strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
import logging

logger = logging.getLogger(__name__)

class FluidStrategy(BaseStrategy):
    """
    Fluid strategy that maintains value ratio toward ideal_ratio with three states.
    
    Parameters:
    - ideal_ratio: Target ratio of token0 to token1 value
    - acceptable_ratio: Acceptable deviation from ideal ratio
    - sprawl_type: Position distribution type (full, dynamic, static)
    - tail_weight: Weight for long tail positions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["ideal_ratio", "acceptable_ratio"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.ideal_ratio = self._get_parameter("ideal_ratio")  # Target ratio
        self.acceptable_ratio = self._get_parameter("acceptable_ratio")  # Acceptable deviation
        self.sprawl_type = self._get_parameter("sprawl_type", "dynamic")
        self.tail_weight = self._get_parameter("tail_weight", 0.2)
        
        # Optional parameters
        self.curve_type = self._get_parameter("curve_type", "gaussian")
        self.curve_params = self._get_parameter("curve_params", {})
        self.max_positions = self._get_parameter("max_positions", 5)
        self.rebalance_threshold = self._get_parameter("rebalance_threshold", 0.1)
        
        # Initialize curve
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Strategy state
        self.current_ratio = None
        self.current_state = "default"  # default, unbalanced, one_sided
        self.last_rebalance_ratio = None
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges based on current ratio and target.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities)
        """
        # Calculate current ratio (assuming we have position data)
        # For now, use a simplified approach
        self.current_ratio = self._estimate_current_ratio(current_price, portfolio_value)
        
        # Determine strategy state
        self.current_state = self._determine_state()
        
        # Calculate target allocation
        target_allocation = self._calculate_target_allocation(current_price, portfolio_value)
        
        # Generate position ranges based on sprawl type
        if self.sprawl_type == "full":
            ranges, liquidities = self._generate_full_sprawl(current_price, portfolio_value, target_allocation)
        elif self.sprawl_type == "dynamic":
            ranges, liquidities = self._generate_dynamic_sprawl(current_price, portfolio_value, target_allocation)
        elif self.sprawl_type == "static":
            ranges, liquidities = self._generate_static_sprawl(current_price, portfolio_value, target_allocation)
        else:
            raise ValueError(f"Unknown sprawl type: {self.sprawl_type}")
        
        # Store rebalance information
        self.last_rebalance_ratio = self.current_ratio
        
        return ranges, liquidities
    
    def _estimate_current_ratio(self, current_price: float, portfolio_value: float) -> float:
        """Estimate current ratio of token0 to token1 value."""
        # This is a simplified estimation
        # In a real implementation, you would use actual position data
        if self.current_ratio is None:
            return self.ideal_ratio
        
        # Simulate some drift
        drift = np.random.normal(0, 0.01)  # Small random drift
        return self.current_ratio * (1 + drift)
    
    def _determine_state(self) -> str:
        """Determine current strategy state."""
        if self.current_ratio is None:
            return "default"
        
        ratio_deviation = abs(self.current_ratio - self.ideal_ratio) / self.ideal_ratio
        
        if ratio_deviation <= self.acceptable_ratio:
            return "default"
        elif ratio_deviation <= self.rebalance_threshold:
            return "unbalanced"
        else:
            return "one_sided"
    
    def _calculate_target_allocation(self, current_price: float, portfolio_value: float) -> Dict[str, float]:
        """Calculate target allocation for tokens."""
        if self.current_state == "default":
            # Maintain ideal ratio
            target_token0_pct = self.ideal_ratio / (1 + self.ideal_ratio)
            target_token1_pct = 1 - target_token0_pct
        elif self.current_state == "unbalanced":
            # Move toward ideal ratio
            current_token0_pct = self.current_ratio / (1 + self.current_ratio)
            ideal_token0_pct = self.ideal_ratio / (1 + self.ideal_ratio)
            
            # Gradual adjustment
            adjustment_factor = 0.5
            target_token0_pct = current_token0_pct + (ideal_token0_pct - current_token0_pct) * adjustment_factor
            target_token1_pct = 1 - target_token0_pct
        else:  # one_sided
            # Aggressive rebalancing
            target_token0_pct = self.ideal_ratio / (1 + self.ideal_ratio)
            target_token1_pct = 1 - target_token0_pct
        
        return {
            "token0_pct": target_token0_pct,
            "token1_pct": target_token1_pct,
            "token0_value": portfolio_value * target_token0_pct,
            "token1_value": portfolio_value * target_token1_pct
        }
    
    def _generate_full_sprawl(
        self,
        current_price: float,
        portfolio_value: float,
        target_allocation: Dict[str, float]
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Generate full sprawl of positions across price range."""
        # Use curve to generate multiple positions
        center_price = current_price
        width_pct = 20.0  # 20% width for full sprawl
        
        distribution = self.curve.generate_distribution(
            center_price, width_pct, portfolio_value * 0.95
        )
        
        ranges = [(lower, upper) for lower, upper, _ in distribution]
        liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def _generate_dynamic_sprawl(
        self,
        current_price: float,
        portfolio_value: float,
        target_allocation: Dict[str, float]
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Generate dynamic sprawl based on current state."""
        if self.current_state == "default":
            # Conservative positioning
            width_pct = 10.0
            max_positions = 3
        elif self.current_state == "unbalanced":
            # Moderate positioning
            width_pct = 15.0
            max_positions = 4
        else:
            # Aggressive positioning
            width_pct = 25.0
            max_positions = 5
        
        # Adjust curve parameters
        curve_params = self.curve_params.copy()
        curve_params["max_bins"] = max_positions
        
        curve = CurveFactory.create_curve(self.curve_type, **curve_params)
        
        distribution = curve.generate_distribution(
            current_price, width_pct, portfolio_value * 0.95
        )
        
        ranges = [(lower, upper) for lower, upper, _ in distribution]
        liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def _generate_static_sprawl(
        self,
        current_price: float,
        portfolio_value: float,
        target_allocation: Dict[str, float]
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Generate static sprawl with fixed parameters."""
        width_pct = 15.0
        max_positions = 4
        
        curve_params = self.curve_params.copy()
        curve_params["max_bins"] = max_positions
        
        curve = CurveFactory.create_curve(self.curve_type, **curve_params)
        
        distribution = curve.generate_distribution(
            current_price, width_pct, portfolio_value * 0.95
        )
        
        ranges = [(lower, upper) for lower, upper, _ in distribution]
        liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def should_rebalance(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> bool:
        """Determine if rebalancing is needed."""
        if not self.initialized:
            return True
        
        # Check if ratio has drifted significantly
        current_ratio = self._estimate_current_ratio(current_price, portfolio_value)
        ratio_deviation = abs(current_ratio - self.ideal_ratio) / self.ideal_ratio
        
        return ratio_deviation > self.acceptable_ratio
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "ideal_ratio": self.ideal_ratio,
            "acceptable_ratio": self.acceptable_ratio,
            "sprawl_type": self.sprawl_type,
            "tail_weight": self.tail_weight,
            "current_ratio": self.current_ratio,
            "current_state": self.current_state,
            "last_rebalance_ratio": self.last_rebalance_ratio,
            "curve_type": self.curve_type,
            "max_positions": self.max_positions
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.current_ratio = None
        self.current_state = "default"
        self.last_rebalance_ratio = None
