"""
Performance metrics calculation for CLMM backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculator for various performance metrics."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_returns(equity_curve: pd.DataFrame) -> pd.Series:
        """
        Calculate returns from equity curve.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            
        Returns:
            Series of returns
        """
        if equity_curve.empty:
            return pd.Series()
        
        return equity_curve["total_value"].pct_change().dropna()
    
    @staticmethod
    def calculate_annualized_return(equity_curve: pd.DataFrame, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            periods_per_year: Number of periods per year (252 for daily, 24 for hourly)
            
        Returns:
            Annualized return as percentage
        """
        if equity_curve.empty:
            return 0.0
        
        total_return = (equity_curve["total_value"].iloc[-1] / equity_curve["total_value"].iloc[0] - 1)
        num_periods = len(equity_curve)
        
        if num_periods <= 1:
            return 0.0
        
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        return annualized_return * 100
    
    @staticmethod
    def calculate_volatility(equity_curve: pd.DataFrame, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility as percentage
        """
        returns = MetricsCalculator.calculate_returns(equity_curve)
        
        if returns.empty:
            return 0.0
        
        volatility = returns.std() * np.sqrt(periods_per_year)
        return volatility * 100
    
    @staticmethod
    def calculate_sharpe_ratio(
        equity_curve: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        returns = MetricsCalculator.calculate_returns(equity_curve)
        
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(
        equity_curve: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        returns = MetricsCalculator.calculate_returns(equity_curve)
        
        if returns.empty:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        
        return sortino_ratio
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.DataFrame) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if equity_curve.empty:
            return 0.0, None, None
        
        # Calculate running maximum
        running_max = equity_curve["total_value"].expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve["total_value"] / running_max - 1) * 100
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        trough_idx = drawdown.idxmin()
        
        # Find peak before trough
        peak_idx = running_max[:trough_idx].idxmax()
        
        return max_drawdown, peak_idx, trough_idx
    
    @staticmethod
    def calculate_calmar_ratio(
        equity_curve: pd.DataFrame,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        annualized_return = MetricsCalculator.calculate_annualized_return(equity_curve, periods_per_year)
        max_drawdown, _, _ = MetricsCalculator.calculate_max_drawdown(equity_curve)
        
        if max_drawdown == 0:
            return 0.0
        
        return abs(annualized_return / max_drawdown)
    
    @staticmethod
    def calculate_win_rate(equity_curve: pd.DataFrame) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            
        Returns:
            Win rate as percentage
        """
        returns = MetricsCalculator.calculate_returns(equity_curve)
        
        if returns.empty:
            return 0.0
        
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        
        return (positive_returns / total_returns) * 100
    
    @staticmethod
    def calculate_profit_factor(equity_curve: pd.DataFrame) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            
        Returns:
            Profit factor
        """
        returns = MetricsCalculator.calculate_returns(equity_curve)
        
        if returns.empty:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_impermanent_loss(
        strategy_equity: pd.DataFrame,
        hodl_equity: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate impermanent loss vs HODL 50:50.
        
        Args:
            strategy_equity: Strategy equity curve
            hodl_equity: HODL 50:50 equity curve
            
        Returns:
            Series of impermanent loss percentages
        """
        if strategy_equity.empty or hodl_equity.empty:
            return pd.Series()
        
        # Align timestamps
        strategy_equity = strategy_equity.set_index("timestamp")
        hodl_equity = hodl_equity.set_index("timestamp")
        
        # Calculate IL
        il = (strategy_equity["total_value"] - hodl_equity["total_value"]) / hodl_equity["total_value"] * 100
        
        return il
    
    @staticmethod
    def calculate_lvr_proxy(
        strategy_equity: pd.DataFrame,
        hodl_equity: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate LVR (Loss-Versus-Rebalancing) proxy.
        
        Args:
            strategy_equity: Strategy equity curve
            hodl_equity: HODL 50:50 equity curve
            
        Returns:
            Series of LVR proxy percentages
        """
        if strategy_equity.empty or hodl_equity.empty:
            return pd.Series()
        
        # Align timestamps
        strategy_equity = strategy_equity.set_index("timestamp")
        hodl_equity = hodl_equity.set_index("timestamp")
        
        # Calculate LVR proxy (simplified)
        # LVR ≈ V_rebal50:50 - V_CLMM_no_fee
        strategy_no_fees = strategy_equity["total_value"] + strategy_equity.get("total_costs", 0)
        lvr = (hodl_equity["total_value"] - strategy_no_fees) / hodl_equity["total_value"] * 100
        
        return lvr
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.DataFrame,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            periods_per_year: Number of periods per year
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of all metrics
        """
        if equity_curve.empty:
            return {}
        
        # Basic metrics
        total_return = (equity_curve["total_value"].iloc[-1] / equity_curve["total_value"].iloc[0] - 1) * 100
        annualized_return = MetricsCalculator.calculate_annualized_return(equity_curve, periods_per_year)
        volatility = MetricsCalculator.calculate_volatility(equity_curve, periods_per_year)
        
        # Risk-adjusted metrics
        sharpe_ratio = MetricsCalculator.calculate_sharpe_ratio(equity_curve, risk_free_rate, periods_per_year)
        sortino_ratio = MetricsCalculator.calculate_sortino_ratio(equity_curve, risk_free_rate, periods_per_year)
        
        # Drawdown metrics
        max_drawdown, peak_date, trough_date = MetricsCalculator.calculate_max_drawdown(equity_curve)
        calmar_ratio = MetricsCalculator.calculate_calmar_ratio(equity_curve, periods_per_year)
        
        # Other metrics
        win_rate = MetricsCalculator.calculate_win_rate(equity_curve)
        profit_factor = MetricsCalculator.calculate_profit_factor(equity_curve)
        
        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown,
            "peak_date": peak_date,
            "trough_date": trough_date,
            "calmar_ratio": calmar_ratio,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_periods": len(equity_curve),
            "final_value": equity_curve["total_value"].iloc[-1],
            "initial_value": equity_curve["total_value"].iloc[0]
        }
    
    @staticmethod
    def calculate_rolling_metrics(
        equity_curve: pd.DataFrame,
        window: int = 252,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: DataFrame with 'total_value' column
            window: Rolling window size
            periods_per_year: Number of periods per year
            
        Returns:
            DataFrame with rolling metrics
        """
        if equity_curve.empty or len(equity_curve) < window:
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(equity_curve)):
            window_data = equity_curve.iloc[i-window:i+1]
            
            metrics = MetricsCalculator.calculate_all_metrics(
                window_data, periods_per_year
            )
            
            metrics["timestamp"] = equity_curve.index[i]
            rolling_data.append(metrics)
        
        return pd.DataFrame(rolling_data)
