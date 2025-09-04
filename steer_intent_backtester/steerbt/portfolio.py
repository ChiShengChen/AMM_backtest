"""
Portfolio accounting module for CLMM backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Position:
    """Represents a CLMM liquidity position."""
    
    def __init__(
        self,
        lower_price: float,
        upper_price: float,
        liquidity: float,
        fee_tier_bps: int = 500,
        created_at: Optional[datetime] = None
    ):
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.liquidity = liquidity
        self.fee_tier_bps = fee_tier_bps
        self.created_at = created_at or datetime.now()
        self.fees_earned = 0.0
        self.last_rebalance_at = None
        
    def get_value(self, current_price: float) -> Tuple[float, float, float]:
        """
        Get current position value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (amount0, amount1, total_value_usd)
        """
        from .uv3_math import calculate_position_value
        
        return calculate_position_value(
            current_price,
            self.lower_price,
            self.upper_price,
            self.liquidity,
            self.fee_tier_bps
        )
    
    def is_in_range(self, price: float) -> bool:
        """Check if price is within position range."""
        return self.lower_price <= price <= self.upper_price
    
    def get_range_width_pct(self) -> float:
        """Get position width as percentage."""
        center = (self.lower_price + self.upper_price) / 2
        return (self.upper_price - self.lower_price) / center * 100
    
    def add_fees(self, fees: float):
        """Add earned fees to position."""
        self.fees_earned += fees
    
    def rebalance(self, new_lower: float, new_upper: float, new_liquidity: float):
        """Rebalance position to new range and liquidity."""
        self.lower_price = new_lower
        self.upper_price = new_upper
        self.liquidity = new_liquidity
        self.last_rebalance_at = datetime.now()


class Portfolio:
    """Portfolio accounting and management."""
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        fee_bps: int = 5,
        slippage_bps: int = 1,
        gas_cost: float = 0.0
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.gas_cost = gas_cost
        
        self.positions: List[Position] = []
        self.transaction_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_gas_paid = 0.0
        self.total_slippage = 0.0
        self.rebalance_count = 0
        
    def add_position(self, position: Position):
        """Add a new position to portfolio."""
        self.positions.append(position)
        
    def remove_position(self, position: Position):
        """Remove a position from portfolio."""
        if position in self.positions:
            self.positions.remove(position)
    
    def get_total_value(self, current_price: float) -> float:
        """
        Get total portfolio value including cash and positions.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total portfolio value in USD
        """
        total = self.cash
        
        for position in self.positions:
            _, _, position_value = position.get_value(current_price)
            total += position_value + position.fees_earned
        
        return total
    
    def get_position_weights(self, current_price: float) -> Dict[int, float]:
        """
        Get current position weights.
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary mapping position index to weight
        """
        total_value = self.get_total_value(current_price)
        weights = {}
        
        for i, position in enumerate(self.positions):
            _, _, position_value = position.get_value(current_price)
            position_total = position_value + position.fees_earned
            weights[i] = position_total / total_value if total_value > 0 else 0
        
        return weights
    
    def rebalance_positions(
        self,
        new_ranges: List[Tuple[float, float]],
        new_liquidities: List[float],
        current_price: float
    ) -> float:
        """
        Rebalance portfolio to new position ranges and liquidities.
        
        Args:
            new_ranges: List of (lower, upper) price ranges
            new_liquidities: List of liquidity amounts
            current_price: Current market price
            
        Returns:
            Total cost of rebalancing
        """
        if len(new_ranges) != len(new_liquidities):
            raise ValueError("Number of ranges must match number of liquidities")
        
        # Calculate current position values
        current_values = []
        for position in self.positions:
            _, _, value = position.get_value(current_price)
            current_values.append(value + position.fees_earned)
        
        # Calculate new position values
        new_values = []
        for (lower, upper), liquidity in zip(new_ranges, new_liquidities):
            from .uv3_math import calculate_position_value
            _, _, value = calculate_position_value(current_price, lower, upper, liquidity)
            new_values.append(value)
        
        # Calculate rebalancing costs
        total_cost = 0.0
        
        # Remove old positions
        for position in self.positions:
            _, _, value = position.get_value(current_price)
            self.cash += value + position.fees_earned
        
        # Clear positions
        self.positions.clear()
        
        # Add new positions
        for (lower, upper), liquidity in zip(new_ranges, new_liquidities):
            new_position = Position(lower, upper, liquidity)
            self.positions.append(new_position)
            
            # Calculate cost to create position
            _, _, value = new_position.get_value(current_price)
            cost = value * (self.fee_bps / 10000.0)  # Trading fees
            total_cost += cost
        
        # Update cash
        self.cash -= total_cost
        
        # Add gas costs
        if self.gas_cost > 0:
            total_cost += self.gas_cost
            self.total_gas_paid += self.gas_cost
        
        # Update transaction history
        self.transaction_history.append({
            "timestamp": datetime.now(),
            "type": "rebalance",
            "cost": total_cost,
            "positions_count": len(self.positions)
        })
        
        self.rebalance_count += 1
        self.total_fees_paid += total_cost
        
        return total_cost
    
    def add_fees_to_positions(self, volume_data: pd.DataFrame, liquidity_share: float):
        """
        Add fees earned by positions based on volume data.
        
        Args:
            volume_data: DataFrame with volume information
            liquidity_share: Share of total liquidity
        """
        for position in self.positions:
            if position.is_in_range(volume_data["close"].iloc[-1]):
                fees = volume_data["quote_volume"].iloc[-1] * liquidity_share * (position.fee_tier_bps / 10000.0)
                position.add_fees(fees)
    
    def record_equity_point(self, timestamp: datetime, current_price: float):
        """Record equity point for performance tracking."""
        total_value = self.get_total_value(current_price)
        
        self.equity_curve.append({
            "timestamp": timestamp,
            "price": current_price,
            "total_value": total_value,
            "cash": self.cash,
            "positions_value": total_value - self.cash,
            "fees_earned": sum(p.fees_earned for p in self.positions),
            "total_costs": self.total_fees_paid + self.total_gas_paid + self.total_slippage
        })
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df["return_pct"] = (df["total_value"] / self.initial_cash - 1) * 100
        df["drawdown"] = (df["total_value"] / df["total_value"].expanding().max() - 1) * 100
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary."""
        if not self.equity_curve:
            return {}
        
        df = self.get_equity_dataframe()
        
        total_return = (df["total_value"].iloc[-1] / self.initial_cash - 1) * 100
        max_drawdown = df["drawdown"].min()
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        returns = df["total_value"].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_fees_paid": self.total_fees_paid,
            "total_gas_paid": self.total_gas_paid,
            "total_slippage": self.total_slippage,
            "rebalance_count": self.rebalance_count,
            "final_value": df["total_value"].iloc[-1],
            "initial_value": self.initial_cash
        }


class BaselinePortfolio:
    """Baseline portfolio for comparison (HODL 50:50, Single Asset)."""
    
    def __init__(self, initial_cash: float = 10000.0, strategy: str = "hodl_50_50"):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.strategy = strategy
        
        # For HODL 50:50
        self.amount0 = 0.0  # Base asset (e.g., ETH)
        self.amount1 = 0.0  # Quote asset (e.g., USDC)
        
        self.equity_curve: List[Dict] = []
        
    def initialize_position(self, initial_price: float):
        """Initialize position with 50:50 allocation."""
        if self.strategy == "hodl_50_50":
            # Split cash 50:50 between assets
            self.amount0 = (self.cash / 2) / initial_price
            self.amount1 = self.cash / 2
            self.cash = 0
        elif self.strategy == "single_asset":
            # 100% in base asset
            self.amount0 = self.cash / initial_price
            self.amount1 = 0
            self.cash = 0
    
    def get_value(self, current_price: float) -> float:
        """Get current portfolio value."""
        return self.amount0 * current_price + self.amount1 + self.cash
    
    def rebalance_to_50_50(self, current_price: float):
        """Rebalance to 50:50 allocation."""
        if self.strategy != "hodl_50_50":
            return
        
        total_value = self.get_value(current_price)
        target_amount0 = total_value / (2 * current_price)
        target_amount1 = total_value / 2
        
        # Calculate rebalancing amounts
        delta_amount0 = target_amount0 - self.amount0
        delta_amount1 = target_amount1 - self.amount1
        
        # Update amounts
        self.amount0 = target_amount0
        self.amount1 = target_amount1
    
    def record_equity_point(self, timestamp: datetime, current_price: float):
        """Record equity point for performance tracking."""
        total_value = self.get_value(current_price)
        
        self.equity_curve.append({
            "timestamp": timestamp,
            "price": current_price,
            "total_value": total_value,
            "amount0": self.amount0,
            "amount1": self.amount1,
            "cash": self.cash
        })
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df["return_pct"] = (df["total_value"] / self.initial_cash - 1) * 100
        df["drawdown"] = (df["total_value"] / df["total_value"].expanding().max() - 1) * 100
        
        return df
