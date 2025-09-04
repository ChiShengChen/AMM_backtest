"""
Main backtesting engine for CLMM strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
import uuid
import json

from .portfolio import Portfolio, BaselinePortfolio
from .strategies import (
    ClassicStrategy, ChannelMultiplierStrategy, BollingerStrategy,
    KeltnerStrategy, DonchianStrategy, StableStrategy, FluidStrategy, ImperfectClassicStrategy
)
from .uv3_math import calculate_fees_earned

logger = logging.getLogger(__name__)

class Backtester:
    """
    Main backtesting engine for CLMM strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        
        # Extract configuration
        self.pair = config["pair"]
        self.interval = config["interval"]
        self.strategy_name = config["strategy"]
        self.strategy_params = config.get("strategy_params", {})
        
        # Portfolio parameters
        self.initial_cash = config.get("initial_cash", 10000.0)
        self.fee_bps = config.get("fee_bps", 5)
        self.slippage_bps = config.get("slippage_bps", 1)
        self.gas_cost = config.get("gas_cost", 0.0)
        self.liq_share = config.get("liq_share", 0.002)
        
        # Data parameters
        self.data_source = config.get("data_source", "binance")
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        
        # Walkforward parameters
        self.walkforward = config.get("walkforward", False)
        self.train_period = config.get("train_period", 180)  # days
        self.validation_period = config.get("validation_period", 60)  # days
        
        # Initialize components
        self.strategy = self._create_strategy()
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            gas_cost=self.gas_cost
        )
        
        # Baseline portfolios
        self.hodl_portfolio = BaselinePortfolio(
            initial_cash=self.initial_cash,
            strategy="hodl_50_50"
        )
        self.single_asset_portfolio = BaselinePortfolio(
            initial_cash=self.initial_cash,
            strategy="single_asset"
        )
        
        # Results storage
        self.results = {
            "run_id": self.run_id,
            "config": config,
            "strategy_info": {},
            "performance": {},
            "baselines": {},
            "transactions": [],
            "equity_curves": {}
        }
        
        logger.info(f"Initialized backtester {self.run_id} for {self.pair} using {self.strategy_name}")
    
    def _create_strategy(self):
        """Create strategy instance based on configuration."""
        strategy_map = {
            "classic": ClassicStrategy,
            "channel_multiplier": ChannelMultiplierStrategy,
            "bollinger": BollingerStrategy,
            "keltner": KeltnerStrategy,
            "donchian": DonchianStrategy,
            "stable": StableStrategy,
            "fluid": FluidStrategy,
            "classic_perfect": ClassicStrategy,
            "classic_slight_imperfect": ClassicStrategy,
            "classic_moderate_imperfect": ClassicStrategy,
            "classic_highly_imperfect": ClassicStrategy,
            "imperfect_classic": ImperfectClassicStrategy
        }
        
        if self.strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        
        strategy_class = strategy_map[self.strategy_name]
        return strategy_class(**self.strategy_params)
    
    def run(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            price_data: Historical price data with OHLCV columns
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest {self.run_id}")
        
        if price_data.empty:
            raise ValueError("Price data is empty")
        
        # Initialize portfolios
        initial_price = price_data["close"].iloc[0]
        self.hodl_portfolio.initialize_position(initial_price)
        self.single_asset_portfolio.initialize_position(initial_price)
        
        # Initialize strategy
        self.strategy.initialize(initial_price, self.initial_cash, price_data)
        
        # Main backtest loop
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            current_price = row["close"]
            current_volume = row.get("volume", 0)
            current_quote_volume = row.get("quote_volume", current_volume * current_price)
            
            # Update baseline portfolios
            self._update_baseline_portfolios(timestamp, current_price, i)
            
            # Check if strategy should rebalance
            should_rebalance = self.strategy.update(
                price_data.iloc[:i+1],
                current_price,
                self.portfolio.get_total_value(current_price)
            )
            
            if should_rebalance:
                # Get new position ranges and liquidities
                ranges, liquidities = self.strategy.calculate_range(
                    price_data.iloc[:i+1],
                    current_price,
                    self.portfolio.get_total_value(current_price)
                )
                
                # Rebalance portfolio
                rebalance_cost = self.portfolio.rebalance_positions(
                    ranges, liquidities, current_price
                )
                
                logger.debug(f"Rebalanced at {timestamp}: {len(ranges)} positions, cost: {rebalance_cost:.2f}")
            
            # Add fees to positions
            if current_quote_volume > 0:
                self.portfolio.add_fees_to_positions(
                    pd.DataFrame([row]), self.liq_share
                )
            
            # Record equity points
            self.portfolio.record_equity_point(timestamp, current_price)
            self.hodl_portfolio.record_equity_point(timestamp, current_price)
            self.single_asset_portfolio.record_equity_point(timestamp, current_price)
        
        # Calculate final results
        self._calculate_results(price_data)
        
        logger.info(f"Completed backtest {self.run_id}")
        return self.results
    
    def _update_baseline_portfolios(self, timestamp: datetime, current_price: float, bar_index: int):
        """Update baseline portfolios."""
        # HODL 50:50 - rebalance daily
        if bar_index % 24 == 0:  # Assuming hourly data
            self.hodl_portfolio.rebalance_to_50_50(current_price)
        
        # Single asset - no rebalancing needed
    
    def _calculate_results(self, price_data: pd.DataFrame):
        """Calculate final backtest results."""
        final_price = price_data["close"].iloc[-1]
        
        # Strategy performance
        strategy_equity = self.portfolio.get_equity_dataframe()
        strategy_performance = self.portfolio.get_performance_summary()
        
        # Baseline performance
        hodl_equity = self.hodl_portfolio.get_equity_dataframe()
        single_asset_equity = self.single_asset_portfolio.get_equity_dataframe()
        
        # Calculate baselines performance
        hodl_performance = self._calculate_baseline_performance(hodl_equity)
        single_asset_performance = self._calculate_baseline_performance(single_asset_equity)
        
        # Calculate impermanent loss and LVR proxy
        il_metrics = self._calculate_impermanent_loss_metrics(
            strategy_equity, hodl_equity, single_asset_equity
        )
        
        # Store results
        self.results.update({
            "strategy_info": self.strategy.get_strategy_info(),
            "performance": strategy_performance,
            "baselines": {
                "hodl_50_50": hodl_performance,
                "single_asset": single_asset_performance
            },
            "equity_curves": {
                "strategy": strategy_equity.to_dict("records"),
                "hodl_50_50": hodl_equity.to_dict("records"),
                "single_asset": single_asset_equity.to_dict("records")
            },
            "impermanent_loss": il_metrics,
            "transactions": self.portfolio.transaction_history,
            "final_price": final_price,
            "total_bars": len(price_data)
        })
    
    def _calculate_baseline_performance(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for baseline portfolio."""
        if equity_df.empty:
            return {}
        
        total_return = (equity_df["total_value"].iloc[-1] / self.initial_cash - 1) * 100
        max_drawdown = equity_df["drawdown"].min()
        
        # Calculate Sharpe ratio
        returns = equity_df["total_value"].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "final_value": equity_df["total_value"].iloc[-1]
        }
    
    def _calculate_impermanent_loss_metrics(
        self,
        strategy_equity: pd.DataFrame,
        hodl_equity: pd.DataFrame,
        single_asset_equity: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate impermanent loss and LVR proxy metrics."""
        if strategy_equity.empty or hodl_equity.empty:
            return {}
        
        # Align timestamps
        strategy_equity = strategy_equity.set_index("timestamp")
        hodl_equity = hodl_equity.set_index("timestamp")
        
        # Calculate IL vs HODL 50:50
        il_series = (strategy_equity["total_value"] - hodl_equity["total_value"]) / hodl_equity["total_value"] * 100
        
        # Calculate LVR proxy (simplified)
        # LVR ≈ V_rebal50:50 - V_CLMM_no_fee
        # For simplicity, we'll use the strategy value without fees
        strategy_no_fees = strategy_equity["total_value"] + strategy_equity.get("total_costs", 0)
        lvr_proxy = (hodl_equity["total_value"] - strategy_no_fees) / hodl_equity["total_value"] * 100
        
        return {
            "impermanent_loss_pct": il_series.to_dict(),
            "lvr_proxy_pct": lvr_proxy.to_dict(),
            "avg_il": il_series.mean(),
            "max_il": il_series.min(),
            "avg_lvr": lvr_proxy.mean(),
            "max_lvr": lvr_proxy.min()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get backtest summary."""
        return {
            "run_id": self.run_id,
            "pair": self.pair,
            "interval": self.interval,
            "strategy": self.strategy_name,
            "period": f"{self.start_date} to {self.end_date}",
            "initial_cash": self.initial_cash,
            "final_value": self.results.get("performance", {}).get("final_value", 0),
            "total_return_pct": self.results.get("performance", {}).get("total_return_pct", 0),
            "max_drawdown_pct": self.results.get("performance", {}).get("max_drawdown_pct", 0),
            "sharpe_ratio": self.results.get("performance", {}).get("sharpe_ratio", 0),
            "rebalance_count": self.results.get("performance", {}).get("rebalance_count", 0),
            "total_fees_paid": self.results.get("performance", {}).get("total_fees_paid", 0)
        }
    
    def save_results(self, filepath: str):
        """Save backtest results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {filepath}")
    
    def run_walkforward(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run walkforward analysis.
        
        Args:
            price_data: Historical price data
            
        Returns:
            List of walkforward results
        """
        if not self.walkforward:
            raise ValueError("Walkforward analysis not enabled")
        
        logger.info("Starting walkforward analysis")
        
        results = []
        current_date = self.start_date
        
        while current_date < self.end_date:
            # Define train and validation periods
            train_end = current_date + timedelta(days=self.train_period)
            validation_end = train_end + timedelta(days=self.validation_period)
            
            if validation_end > self.end_date:
                break
            
            # Split data
            train_data = price_data[
                (price_data.index >= current_date) & (price_data.index < train_end)
            ]
            validation_data = price_data[
                (price_data.index >= train_end) & (price_data.index < validation_end)
            ]
            
            # Run backtest on training data
            train_results = self._run_period(train_data, f"train_{current_date.date()}")
            
            # Run backtest on validation data
            validation_results = self._run_period(validation_data, f"validation_{current_date.date()}")
            
            # Store results
            period_results = {
                "period": f"{current_date.date()} to {validation_end.date()}",
                "train": train_results,
                "validation": validation_results
            }
            results.append(period_results)
            
            # Move to next period
            current_date = train_end
        
        logger.info(f"Completed walkforward analysis with {len(results)} periods")
        return results
    
    def _run_period(self, price_data: pd.DataFrame, period_name: str) -> Dict[str, Any]:
        """Run backtest for a specific period."""
        # Create a copy of the backtester for this period
        period_backtester = Backtester(self.config)
        period_backtester.run_id = f"{self.run_id}_{period_name}"
        
        # Run backtest
        results = period_backtester.run(price_data)
        
        return results.get("summary", {})
