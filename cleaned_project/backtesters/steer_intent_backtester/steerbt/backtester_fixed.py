"""
Fixed Backtester implementation for CLMM strategies.
修復版本的Backtester實現 - 使用修復後的Portfolio
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import uuid

from .portfolio_fixed import Portfolio
from .strategies import get_strategy_class

logger = logging.getLogger(__name__)

class Backtester:
    """Fixed backtester for CLMM strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        
        # Initialize strategy
        strategy_name = config.get('strategy', 'classic')
        strategy_class = get_strategy_class(strategy_name)
        strategy_params = config.get('strategy_params', {})
        self.strategy = strategy_class(**strategy_params)
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_cash=config.get('initial_cash', 10000.0),
            fee_bps=config.get('fee_bps', 5),
            slippage_bps=config.get('slippage_bps', 1),
            gas_cost=config.get('gas_cost', 0.0)
        )
        
        # Initialize baseline portfolios for comparison
        self.hodl_portfolio = Portfolio(
            initial_cash=config.get('initial_cash', 10000.0),
            fee_bps=0,  # No fees for HODL
            slippage_bps=0,
            gas_cost=0.0
        )
        
        self.single_asset_portfolio = Portfolio(
            initial_cash=config.get('initial_cash', 10000.0),
            fee_bps=0,  # No fees for single asset
            slippage_bps=0,
            gas_cost=0.0
        )
        
        self.results = None
        
        # Strategy parameters
        self.liq_share = config.get('liq_share', 0.001)
        
        logger.info(f"Initialized backtester {self.run_id} for {config.get('pair', 'UNKNOWN')} using {strategy_name}")
    
    def run(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run the backtest."""
        logger.info(f"Starting backtest {self.run_id}")
        
        try:
            # Initialize strategy
            initial_price = data['close'].iloc[0]
            initial_portfolio_value = self.portfolio.get_total_value(initial_price)
            self.strategy.initialize(initial_price, initial_portfolio_value, data)
            
            # Run main backtest loop
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_price = row['close']
                
                # Update baseline portfolios
                self._update_baseline_portfolios(timestamp, current_price, i)
                
            # Check if strategy should rebalance (only if we have cash)
            if self.portfolio.cash > 0:
                should_rebalance = self.strategy.update(
                    data.iloc[:i+1],
                    current_price,
                    self.portfolio.get_total_value(current_price)
                )
                
                if should_rebalance:
                    # Get new position ranges and liquidities
                    ranges, liquidities = self.strategy.calculate_range(
                        data.iloc[:i+1],
                        current_price,
                        self.portfolio.get_total_value(current_price)
                    )
                    
                    # Rebalance portfolio
                    rebalance_cost = self.portfolio.rebalance_positions(
                        ranges, liquidities, current_price
                    )
                    
                    logger.debug(f"Rebalanced at {timestamp}: {len(ranges)} positions, cost: {rebalance_cost:.2f}")
            else:
                # No cash left, skip rebalancing
                pass
                
                # Add fees to positions
                if 'volume' in row and row['volume'] > 0:
                    self.portfolio.add_fees_to_positions(
                        pd.DataFrame([row]), self.liq_share
                    )
                
                # Record equity points
                self.portfolio.record_equity_point(timestamp, current_price)
                self.hodl_portfolio.record_equity_point(timestamp, current_price)
                self.single_asset_portfolio.record_equity_point(timestamp, current_price)
            
            # Calculate final results
            self._calculate_results(data)
            
            logger.info(f"Completed backtest {self.run_id}")
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest {self.run_id} failed: {e}")
            return None
    
    def _update_baseline_portfolios(self, timestamp: datetime, current_price: float, bar_index: int):
        """Update baseline portfolios."""
        # HODL 50:50 - rebalance daily
        if bar_index % 24 == 0:  # Assuming hourly data
            # Simple HODL 50:50 implementation
            total_value = self.hodl_portfolio.get_total_value(current_price)
            self.hodl_portfolio.cash = total_value / 2
            # Note: In a real implementation, we'd track token amounts
        
        # Single asset - no rebalancing needed
    
    def _calculate_results(self, data: pd.DataFrame):
        """Calculate final backtest results."""
        # Get equity curves
        strategy_equity = self.portfolio.get_equity_curve()
        hodl_equity = self.hodl_portfolio.get_equity_curve()
        single_asset_equity = self.single_asset_portfolio.get_equity_curve()
        
        # Calculate metrics
        strategy_stats = self.portfolio.get_summary_stats()
        hodl_stats = self.hodl_portfolio.get_summary_stats()
        single_asset_stats = self.single_asset_portfolio.get_summary_stats()
        
        self.results = {
            'strategy_name': self.config.get('strategy', 'unknown'),
            'run_id': self.run_id,
            'equity_curves': {
                'strategy': strategy_equity.to_dict('records') if not strategy_equity.empty else [],
                'hodl': hodl_equity.to_dict('records') if not hodl_equity.empty else [],
                'single_asset': single_asset_equity.to_dict('records') if not single_asset_equity.empty else []
            },
            'metrics': {
                'strategy': strategy_stats,
                'hodl': hodl_stats,
                'single_asset': single_asset_stats
            },
            'config': self.config
        }
