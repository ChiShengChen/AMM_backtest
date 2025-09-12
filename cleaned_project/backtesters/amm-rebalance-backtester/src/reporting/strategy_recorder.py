"""
Strategy parameter and methodology recorder for AMM backtester.
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StrategyRecorder:
    """Record and save strategy parameters and methodologies."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy definitions
        self.strategy_definitions = {
            'Baseline-Static': {
                'type': 'Passive',
                'description': 'Ultra-wide passive positions with minimal rebalancing',
                'methodology': {
                    'position_width': 'Fixed ultra-wide (500% of price)',
                    'rebalancing_trigger': 'Only when price moves outside range or cooldown expires',
                    'rebalancing_frequency': 'Very low (weekly/monthly)',
                    'liquidity_concentration': 'Single wide position',
                    'risk_management': 'Passive, relies on wide range to avoid IL',
                    'fee_capture': 'Low, due to wide positions',
                    'impermanent_loss_risk': 'High, due to wide positions'
                },
                'parameters': {
                    'width_pct': '500.0% (configurable)',
                    'rebalance_cooldown_hours': '168 (1 week)',
                    'position_count': '1',
                    'target_skew': '50/50 (maintained passively)'
                },
                'use_cases': [
                    'Long-term passive liquidity provision',
                    'Low-maintenance strategies',
                    'Benchmark for comparison'
                ],
                'pros': [
                    'Minimal gas costs',
                    'Low maintenance',
                    'Stable positions'
                ],
                'cons': [
                    'High IL risk',
                    'Low fee capture',
                    'Poor capital efficiency'
                ]
            },
            
            'Baseline-Fixed': {
                'type': 'Traditional',
                'description': 'Fixed width positions with fixed price deviation triggers',
                'methodology': {
                    'position_width': 'Fixed medium width (50% of price)',
                    'rebalancing_trigger': 'Price deviation threshold + cooldown timer',
                    'rebalancing_frequency': 'Medium (daily/weekly)',
                    'liquidity_concentration': 'Single medium position',
                    'risk_management': 'Fixed thresholds, predictable behavior',
                    'fee_capture': 'Medium, balanced approach',
                    'impermanent_loss_risk': 'Medium, controlled by fixed width'
                },
                'parameters': {
                    'width_pct': '50.0% (configurable)',
                    'price_deviation_bps': '50 bps (0.5%)',
                    'rebalance_cooldown_hours': '24 (1 day)',
                    'position_count': '1',
                    'target_skew': '50/50 (rebalanced to target)'
                },
                'use_cases': [
                    'Traditional platform strategies',
                    'Balanced risk-return approach',
                    'Predictable behavior'
                ],
                'pros': [
                    'Predictable behavior',
                    'Balanced risk-return',
                    'Medium fee capture'
                ],
                'cons': [
                    'Not adaptive to market conditions',
                    'Fixed thresholds may be suboptimal',
                    'Medium gas costs'
                ]
            },
            
            'Dynamic-Vol': {
                'type': 'Adaptive',
                'description': 'Volatility-adaptive width with price deviation triggers',
                'methodology': {
                    'position_width': 'Dynamic based on volatility: W_t = k × σ_t',
                    'rebalancing_trigger': 'Price deviation + volatility-adjusted cooldown',
                    'rebalancing_frequency': 'High (daily/hourly)',
                    'liquidity_concentration': 'Single adaptive position',
                    'risk_management': 'Volatility-based width adjustment',
                    'fee_capture': 'High, due to adaptive positioning',
                    'impermanent_loss_risk': 'Low, due to adaptive width'
                },
                'parameters': {
                    'vol_estimator': 'EWMA or Range-based (configurable)',
                    'k_width': '1.5 (configurable, 0.8-2.0)',
                    'price_deviation_bps': '50 bps (configurable, 20-120)',
                    'rebalance_cooldown_hours': '24 (configurable, 6-48)',
                    'vol_window_hours': '168 (1 week)',
                    'position_count': '1',
                    'target_skew': '50/50 (rebalanced to target)'
                },
                'use_cases': [
                    'Volatile market conditions',
                    'High-performance strategies',
                    'Adaptive risk management'
                ],
                'pros': [
                    'Adaptive to market conditions',
                    'High fee capture',
                    'Low IL risk',
                    'Optimal APR/MDD balance'
                ],
                'cons': [
                    'Higher gas costs',
                    'More complex',
                    'Requires parameter tuning'
                ]
            },
            
            'Dynamic-Inventory': {
                'type': 'Smart',
                'description': 'Inventory skew + fee density triggers with low-frequency reinvestment',
                'methodology': {
                    'position_width': 'Dynamic based on inventory skew and fee density',
                    'rebalancing_trigger': 'Inventory skew threshold + fee density + cooldown',
                    'rebalancing_frequency': 'Medium-high (daily)',
                    'liquidity_concentration': 'Single smart position',
                    'risk_management': 'Inventory-based + fee density monitoring',
                    'fee_capture': 'Very high, due to smart positioning',
                    'impermanent_loss_risk': 'Very low, due to smart management'
                },
                'parameters': {
                    'skew_threshold_pct': '15.0% (configurable, 5-25%)',
                    'fee_density_window_h': '24 (configurable, 6-48)',
                    'reinvest_frequency_h': '48 (configurable, 12-72)',
                    'target_skew': '0.5 (50/50)',
                    'base_width_pct': '30.0%',
                    'position_count': '1'
                },
                'use_cases': [
                    'High-frequency trading environments',
                    'Smart liquidity management',
                    'Lowest MDD requirements'
                ],
                'pros': [
                    'Lowest MDD',
                    'Highest fee capture',
                    'Smart inventory management',
                    'Low IL risk'
                ],
                'cons': [
                    'Highest gas costs',
                    'Most complex',
                    'Requires sophisticated monitoring'
                ]
            }
        }
    
    def record_strategy_parameters(self, results: Dict[str, Any], study_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record strategy parameters and performance."""
        timestamp = datetime.now().isoformat()
        
        # Extract strategy performance
        summary_df = results['summary']
        strategies = summary_df['strategy'].tolist()
        
        # Create strategy record
        strategy_record = {
            'timestamp': timestamp,
            'experiment_info': {
                'total_strategies': len(strategies),
                'best_apr': summary_df['apr'].max(),
                'best_strategy': summary_df.loc[summary_df['apr'].idxmax(), 'strategy'],
                'lowest_mdd': summary_df['mdd'].min(),
                'lowest_mdd_strategy': summary_df.loc[summary_df['mdd'].idxmin(), 'strategy'],
                'best_sharpe': summary_df['sharpe'].max(),
                'best_sharpe_strategy': summary_df.loc[summary_df['sharpe'].idxmax(), 'strategy']
            },
            'strategies': {}
        }
        
        # Record each strategy
        for strategy_name in strategies:
            strategy_data = summary_df[summary_df['strategy'] == strategy_name].iloc[0]
            
            # Get strategy definition
            strategy_def = self.strategy_definitions.get(strategy_name, {})
            
            # Record strategy details
            strategy_record['strategies'][strategy_name] = {
                'performance': {
                    'apr': float(strategy_data['apr']),
                    'mdd': float(strategy_data['mdd']),
                    'sharpe': float(strategy_data['sharpe']),
                    'calmar': float(strategy_data.get('calmar', 0)),
                    'rebalance_count': int(strategy_data['rebalance_count'])
                },
                'definition': strategy_def,
                'parameters': self._extract_strategy_parameters(strategy_name, study_params)
            }
        
        # Add optimization results if available
        if study_params:
            strategy_record['optimization'] = {
                'best_parameters': study_params,
                'optimization_method': 'Optuna',
                'objective': 'Maximize APR'
            }
        
        return strategy_record
    
    def _extract_strategy_parameters(self, strategy_name: str, study_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract parameters for a specific strategy."""
        if strategy_name == 'Dynamic-Vol' and study_params:
            return {
                'k_width': study_params.get('k_width', 1.5),
                'price_deviation_bps': study_params.get('price_deviation_bps', 50.0),
                'rebalance_cooldown_hours': study_params.get('rebalance_cooldown_hours', 24),
                'vol_estimator': 'ewma',
                'vol_window_hours': 168
            }
        elif strategy_name == 'Dynamic-Inventory':
            return {
                'skew_threshold_pct': 15.0,
                'fee_density_window_h': 24,
                'reinvest_frequency_h': 48,
                'target_skew': 0.5,
                'base_width_pct': 30.0
            }
        elif strategy_name == 'Baseline-Fixed':
            return {
                'width_pct': 50.0,
                'price_deviation_bps': 50.0,
                'rebalance_cooldown_hours': 24
            }
        elif strategy_name == 'Baseline-Static':
            return {
                'width_pct': 500.0,
                'rebalance_cooldown_hours': 168
            }
        else:
            return {}
    
    def save_strategy_record(self, strategy_record: Dict[str, Any], filename: str = None) -> str:
        """Save strategy record to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_record_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(strategy_record, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Strategy record saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save strategy record: {e}")
            raise
    
    def save_strategy_summary_csv(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save strategy summary to CSV with methodology."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_summary_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        try:
            # Create enhanced summary
            summary_df = results['summary'].copy()
            
            # Add methodology columns
            summary_df['strategy_type'] = summary_df['strategy'].map(
                lambda x: self.strategy_definitions.get(x, {}).get('type', 'Unknown')
            )
            summary_df['description'] = summary_df['strategy'].map(
                lambda x: self.strategy_definitions.get(x, {}).get('description', '')
            )
            summary_df['rebalancing_frequency'] = summary_df['strategy'].map(
                lambda x: self.strategy_definitions.get(x, {}).get('methodology', {}).get('rebalancing_frequency', '')
            )
            summary_df['il_risk'] = summary_df['strategy'].map(
                lambda x: self.strategy_definitions.get(x, {}).get('methodology', {}).get('impermanent_loss_risk', '')
            )
            summary_df['fee_capture'] = summary_df['strategy'].map(
                lambda x: self.strategy_definitions.get(x, {}).get('methodology', {}).get('fee_capture', '')
            )
            
            # Save to CSV
            summary_df.to_csv(filepath, index=False)
            
            logger.info(f"Strategy summary saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save strategy summary: {e}")
            raise
    
    def generate_strategy_report(self, results: Dict[str, Any], study_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive strategy report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_report_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("AMM Dynamic Rebalancing Strategy Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Performance Summary
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 30 + "\n")
                summary_df = results['summary']
                for _, row in summary_df.iterrows():
                    f.write(f"{row['strategy']}:\n")
                    f.write(f"  APR: {row['apr']:.1f}%\n")
                    f.write(f"  MDD: {row['mdd']:.1f}%\n")
                    f.write(f"  Sharpe: {row['sharpe']:.2f}\n")
                    if 'calmar' in row:
                        f.write(f"  Calmar: {row['calmar']:.2f}\n")
                    f.write(f"  Rebalances: {row['rebalance_count']}\n\n")
                
                # Strategy Analysis
                f.write("STRATEGY ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for strategy_name in summary_df['strategy']:
                    if strategy_name in self.strategy_definitions:
                        def_info = self.strategy_definitions[strategy_name]
                        f.write(f"{strategy_name} ({def_info.get('type', 'Unknown')}):\n")
                        f.write(f"  {def_info.get('description', '')}\n")
                        f.write(f"  Rebalancing: {def_info.get('methodology', {}).get('rebalancing_frequency', '')}\n")
                        f.write(f"  IL Risk: {def_info.get('methodology', {}).get('impermanent_loss_risk', '')}\n")
                        f.write(f"  Fee Capture: {def_info.get('methodology', {}).get('fee_capture', '')}\n\n")
                
                # Optimization Results
                if study_params:
                    f.write("OPTIMIZATION RESULTS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Best Parameters:\n")
                    for param, value in study_params.items():
                        f.write(f"  {param}: {value}\n")
                    f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                best_apr = summary_df.loc[summary_df['apr'].idxmax(), 'strategy']
                lowest_mdd = summary_df.loc[summary_df['mdd'].idxmin(), 'strategy']
                best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax(), 'strategy']
                
                f.write(f"Best APR Strategy: {best_apr}\n")
                f.write(f"Lowest MDD Strategy: {lowest_mdd}\n")
                f.write(f"Best Risk-Adjusted: {best_sharpe}\n\n")
                
                f.write("For high-performance: Use Dynamic-Vol or Dynamic-Inventory\n")
                f.write("For low-maintenance: Use Baseline-Static\n")
                f.write("For balanced approach: Use Baseline-Fixed\n")
            
            logger.info(f"Strategy report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate strategy report: {e}")
            raise
