"""
Fixed Portfolio implementation for CLMM strategies.
修復版本的Portfolio實現 - 正確的AMM邏輯
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging

from .uv3_math_stable import calculate_position_value

logger = logging.getLogger(__name__)

class Position:
    """Represents a single CLMM position."""
    
    def __init__(self, lower_price: float, upper_price: float, liquidity: float):
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.liquidity = liquidity
        self.fees_earned = 0.0
        
    def get_value(self, current_price: float) -> Tuple[float, float, float]:
        """Get position value in terms of token0, token1, and total USD value."""
        amount0, amount1, total_value = calculate_position_value(
            current_price, self.lower_price, self.upper_price, self.liquidity
        )
        return amount0, amount1, total_value

class Portfolio:
    """Portfolio management for CLMM strategies with fixed AMM logic."""
    
    def __init__(
        self,
        initial_cash: float,
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
        self.equity_curve: List[Dict] = []
        self.transaction_history: List[Dict] = []
        
        # Track costs and fees
        self.total_fees_paid = 0.0
        self.total_gas_paid = 0.0
        self.total_slippage_paid = 0.0
        
    def get_total_value(self, current_price: float) -> float:
        """Get total portfolio value."""
        total = self.cash
        
        for position in self.positions:
            _, _, value = position.get_value(current_price)
            total += value + position.fees_earned
            
        return total
    
    def rebalance_positions(
        self,
        new_ranges: List[Tuple[float, float]],
        new_liquidities: List[float],
        current_price: float
    ) -> float:
        """
        Rebalance portfolio to new position ranges and liquidities.
        修復：正確的AMM rebalance邏輯，防止現金耗盡
        """
        if len(new_ranges) != len(new_liquidities):
            raise ValueError("Number of ranges must match number of liquidities")
        
        # 計算當前持倉的總價值
        current_total_value = 0.0
        for position in self.positions:
            _, _, value = position.get_value(current_price)
            current_total_value += value + position.fees_earned
        
        # 計算新持倉的總價值
        new_total_value = 0.0
        for (lower, upper), liquidity in zip(new_ranges, new_liquidities):
            _, _, value = calculate_position_value(current_price, lower, upper, liquidity)
            new_total_value += value
        
        # 修復：正確計算重新平衡成本
        # 只收取手續費，不收取價值差額
        total_position_value = max(current_total_value, new_total_value)
        
        # 限制手續費，避免過度消耗現金
        max_fee_ratio = 0.01  # 最多收取1%的手續費
        rebalance_cost = min(
            total_position_value * (self.fee_bps / 10000.0),
            total_position_value * max_fee_ratio
        )
        
        # 進一步限制：手續費不應超過現金的50%
        max_fee_from_cash = self.cash * 0.5
        rebalance_cost = min(rebalance_cost, max_fee_from_cash)
        
        # 檢查現金是否足夠支付手續費
        if self.cash < rebalance_cost:
            # 如果現金不足，調整持倉規模以匹配可用現金
            if self.cash > 0:
                scale_factor = self.cash / rebalance_cost
                new_liquidities = [liq * scale_factor for liq in new_liquidities]
                rebalance_cost = self.cash
            else:
                # 現金完全耗盡，停止重新平衡
                new_liquidities = [0.0] * len(new_liquidities)
                rebalance_cost = 0.0
        
        # 更新現金（只扣除手續費）
        self.cash -= rebalance_cost
        self.total_fees_paid += rebalance_cost
        
        # 清除舊持倉
        self.positions.clear()
        
        # 添加新持倉
        for (lower, upper), liquidity in zip(new_ranges, new_liquidities):
            if liquidity > 0:  # 只添加有效的持倉
                new_position = Position(lower, upper, liquidity)
                self.positions.append(new_position)
        
        # 添加gas成本
        if self.gas_cost > 0 and self.cash >= self.gas_cost:
            self.cash -= self.gas_cost
            self.total_gas_paid += self.gas_cost
            rebalance_cost += self.gas_cost
        
        # 記錄交易歷史
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'type': 'rebalance',
            'old_positions': len(self.positions),
            'new_positions': len(new_ranges),
            'cost': rebalance_cost,
            'cash_after': self.cash
        })
        
        return rebalance_cost
    
    def add_fees_to_positions(
        self,
        volume_data: pd.DataFrame,
        liquidity_share: float
    ):
        """Add fees earned to positions based on trading volume."""
        if volume_data.empty:
            return
        
        # 計算總交易量
        total_volume = volume_data['volume'].sum() if 'volume' in volume_data.columns else 0
        
        if total_volume > 0:
            # 計算手續費收入
            fee_rate = self.fee_bps / 10000.0
            total_fees = total_volume * liquidity_share * fee_rate
            
            # 將手續費分配給持倉
            if self.positions:
                fees_per_position = total_fees / len(self.positions)
                for position in self.positions:
                    position.fees_earned += fees_per_position
    
    def record_equity_point(self, timestamp: datetime, current_price: float):
        """Record current portfolio state."""
        total_value = self.get_total_value(current_price)
        
        # 計算持倉總價值
        positions_value = 0.0
        for position in self.positions:
            _, _, value = position.get_value(current_price)
            positions_value += value + position.fees_earned
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'price': current_price,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'fees_earned': sum(p.fees_earned for p in self.positions),
            'total_costs': self.total_fees_paid + self.total_gas_paid + self.total_slippage_paid
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_returns(self) -> pd.Series:
        """Calculate returns from equity curve."""
        df = self.get_equity_curve()
        if df.empty:
            return pd.Series()
        
        returns = df["total_value"].pct_change().dropna()
        return returns
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        df = self.get_equity_curve()
        if df.empty:
            return {}
        
        returns = self.calculate_returns()
        
        if len(returns) == 0:
            return {}
        
        # 計算年化回報率
        initial_value = df["total_value"].iloc[0]
        final_value = df["total_value"].iloc[-1]
        
        if initial_value <= 0:
            total_return = -1.0  # 完全虧損
        else:
            total_return = (final_value - initial_value) / initial_value
        
        years = (df.index[-1] - df.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 計算年化波動率
        annual_volatility = returns.std() * np.sqrt(365.25 * 24)  # 假設小時數據
        
        # 計算夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 計算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        
        # 避免除零錯誤
        drawdown = np.where(running_max > 0, (cumulative - running_max) / running_max, 0)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': df["total_value"].iloc[-1],
            'total_fees_paid': self.total_fees_paid,
            'total_gas_paid': self.total_gas_paid,
            'rebalance_count': len(self.transaction_history)
        }
