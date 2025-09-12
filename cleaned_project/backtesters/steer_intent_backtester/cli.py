#!/usr/bin/env python3
"""
Command-line interface for Steer Intent Backtester.
"""

import click
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from steerbt.data.binance import BinanceDataFetcher
from steerbt.data.kraken import KrakenDataFetcher
from steerbt.backtester import Backtester
from steerbt.reports import ReportGenerator
from steerbt.ml import SteerFeatureEngineer, SteerMLTrainer, SteerMLStrategy, IntentPredictor, PricePredictor
from steerbt.ml.models import SteerMLBollingerStrategy
from steerbt.strategies import MLBollingerStrategy, MLKeltnerStrategy, MLDonchianStrategy, MLHybridStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def main():
    """Steer Intent Backtester - CLMM backtesting system."""
    pass

@main.command()
@click.option('--source', type=click.Choice(['binance', 'kraken']), default='binance', help='Data source')
@click.option('--symbol', required=True, help='Trading pair symbol (e.g., ETHUSDC)')
@click.option('--interval', type=click.Choice(['1h', '1d', '4h', '15m', '1m']), default='1h', help='Time interval')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD, defaults to today)')
@click.option('--out', required=True, help='Output CSV file path')
@click.option('--limit', default=1000, help='Maximum records per request')
def fetch(source, symbol, interval, start, end, out, limit):
    """Fetch historical data from specified source."""
    try:
        # Parse dates
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d') if end else datetime.now()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(out), exist_ok=True)
        
        if source == 'binance':
            fetcher = BinanceDataFetcher()
            data = fetcher.fetch_klines(symbol, interval, start_date, end_date, limit)
        elif source == 'kraken':
            fetcher = KrakenDataFetcher()
            # Map symbol for Kraken
            kraken_symbol = symbol.replace('USDC', '/USDC').replace('USDT', '/USDT')
            data = fetcher.fetch_ohlc(kraken_symbol, interval, start_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        if data.empty:
            click.echo("No data fetched")
            return
        
        # Save to CSV
        data.to_csv(out, index=False)
        click.echo(f"Fetched {len(data)} records from {start_date.date()} to {end_date.date()}")
        click.echo(f"Data saved to: {out}")
        
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)
        raise click.Abort()

@main.command()
@click.option('--pair', required=True, help='Trading pair (e.g., ETHUSDC)')
@click.option('--interval', type=click.Choice(['1h', '1d']), default='1h', help='Time interval')
@click.option('--strategy', type=click.Choice([
    'classic', 'channel_multiplier', 'bollinger', 'keltner', 
    'donchian', 'stable', 'fluid'
]), required=True, help='Strategy to use')
@click.option('--data-file', help='CSV file with historical data')
@click.option('--start-date', help='Start date for backtest (YYYY-MM-DD)')
@click.option('--end-date', help='End date for backtest (YYYY-MM-DD)')
@click.option('--initial-cash', default=10000.0, help='Initial capital')
@click.option('--fee-bps', default=5, help='Trading fees in basis points')
@click.option('--slippage-bps', default=1, help='Slippage in basis points')
@click.option('--gas-cost', default=0.0, help='Gas cost per rebalance')
@click.option('--liq-share', default=0.002, help='Liquidity share')
@click.option('--walkforward', is_flag=True, help='Enable walkforward analysis')
@click.option('--output-dir', default='reports', help='Output directory for reports')
def backtest(pair, interval, strategy, data_file, start_date, end_date, initial_cash, 
             fee_bps, slippage_bps, gas_cost, liq_share, walkforward, output_dir):
    """Run backtest with specified strategy."""
    try:
        # Load data
        if data_file:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
        else:
            # Fetch data from Binance
            click.echo("Fetching data from Binance...")
            fetcher = BinanceDataFetcher()
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=30)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            data = fetcher.fetch_klines(pair, interval, start_dt, end_dt)
            data = data.set_index('timestamp')
        
        if data.empty:
            click.echo("No data available for backtest")
            return
        
        # Prepare strategy parameters
        strategy_params = _get_strategy_params(strategy)
        
        # Create backtest configuration
        config = {
            "pair": pair,
            "interval": interval,
            "strategy": strategy,
            "strategy_params": strategy_params,
            "initial_cash": initial_cash,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "gas_cost": gas_cost,
            "liq_share": liq_share,
            "walkforward": walkforward,
            "start_date": data.index[0],
            "end_date": data.index[-1]
        }
        
        # Run backtest
        click.echo(f"Running backtest for {pair} using {strategy} strategy...")
        backtester = Backtester(config)
        
        if walkforward:
            results = backtester.run_walkforward(data)
            click.echo(f"Walkforward analysis completed with {len(results)} periods")
        else:
            results = backtester.run(data)
            click.echo("Backtest completed successfully")
        
        # Generate reports
        click.echo("Generating reports...")
        report_files = backtester.generate_reports(output_dir)
        
        click.echo(f"Reports generated successfully!")
        click.echo(f"Experiment directory: {report_files.get('index', '').replace('/index_', '/').replace('_', '/')}")
        click.echo(f"Open index file to view all results: {report_files.get('index', '')}")
        
        for report_type, filepath in report_files.items():
            if filepath:  # Skip empty filepaths
                click.echo(f"  {report_type}: {filepath}")
        
        # Show summary
        if not walkforward:
            summary = backtester.get_summary()
            click.echo("\nBacktest Summary:")
            click.echo(f"  Total Return: {summary.get('total_return_pct', 0):.2f}%")
            click.echo(f"  Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
            click.echo(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            click.echo(f"  Rebalance Count: {summary.get('rebalance_count', 0)}")
        
    except Exception as e:
        click.echo(f"Error running backtest: {e}", err=True)
        raise click.Abort()

@main.command()
@click.option('--run-id', required=True, help='Backtest run ID')
@click.option('--output-dir', default='reports', help='Output directory for reports')
def report(run_id, output_dir):
    """Generate reports for a completed backtest."""
    try:
        # Look for results file
        results_file = os.path.join(output_dir, f"results_{run_id}.json")
        
        if not os.path.exists(results_file):
            click.echo(f"Results file not found: {results_file}")
            click.echo("Please run a backtest first to generate results")
            return
        
        # Load results and generate reports
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        click.echo(f"Generating reports for run {run_id}...")
        report_generator = ReportGenerator(results, output_dir)
        report_files = report_generator.generate_all_reports()
        
        click.echo(f"Reports generated successfully!")
        click.echo(f"Experiment directory: {report_generator.experiment_dir}")
        click.echo(f"Open index file to view all results: {report_files.get('index', '')}")
        
        for report_type, filepath in report_files.items():
            if filepath:
                click.echo(f"  {report_type}: {filepath}")
        
    except Exception as e:
        click.echo(f"Error generating reports: {e}", err=True)
        raise click.Abort()

def _get_strategy_params(strategy: str) -> dict:
    """Get default parameters for a strategy."""
    if strategy == 'classic':
        return {
            "width_mode": "percent",
            "width_value": 10.0,
            "placement_mode": "center",
            "curve_type": "uniform"
        }
    elif strategy == 'channel_multiplier':
        return {"width_pct": 10.0}
    elif strategy == 'bollinger':
        return {"n": 20, "k": 2.0}
    elif strategy == 'keltner':
        return {"n": 20, "m": 2.0}
    elif strategy == 'donchian':
        return {"n": 20, "width_multiplier": 1.0}
    elif strategy == 'stable':
        return {
            "peg_method": "sma",
            "peg_period": 20,
            "width_pct": 15.0,
            "curve_type": "gaussian",
            "bin_count": 5
        }
    elif strategy == 'fluid':
        return {
            "ideal_ratio": 1.0,
            "acceptable_ratio": 0.1,
            "sprawl_type": "dynamic",
            "tail_weight": 0.2
        }
    else:
        return {}

@main.command()
def strategies():
    """List available strategies and their parameters."""
    strategies_info = {
        'classic': {
            'description': 'Classic rebalancing with configurable width and placement',
            'params': ['width_mode', 'width_value', 'placement_mode', 'curve_type']
        },
        'channel_multiplier': {
            'description': 'Single symmetric percentage width around price',
            'params': ['width_pct']
        },
        'bollinger': {
            'description': 'Bollinger Bands: SMA ± k*Std',
            'params': ['n', 'k']
        },
        'keltner': {
            'description': 'Keltner Channels: EMA ± m*ATR',
            'params': ['n', 'm']
        },
        'donchian': {
            'description': 'Donchian Channels: HH/LL over N periods',
            'params': ['n', 'width_multiplier']
        },
        'stable': {
            'description': 'Multi-position around computed peg',
            'params': ['peg_method', 'peg_period', 'width_pct', 'curve_type', 'bin_count']
        },
        'fluid': {
            'description': 'Maintain value ratio toward ideal_ratio',
            'params': ['ideal_ratio', 'acceptable_ratio', 'sprawl_type', 'tail_weight']
        }
    }
    
    click.echo("Available Strategies:")
    click.echo("=" * 50)
    
    for name, info in strategies_info.items():
        click.echo(f"\n{name.upper()}")
        click.echo(f"  Description: {info['description']}")
        click.echo(f"  Parameters: {', '.join(info['params'])}")

@main.command()
def curves():
    """List available liquidity distribution curves."""
    from steerbt.curves import CurveFactory
    
    curves = CurveFactory.get_available_curves()
    click.echo("Available Liquidity Distribution Curves:")
    click.echo("=" * 40)
    
    for curve in curves:
        params = CurveFactory.get_curve_params(curve)
        click.echo(f"\n{curve.upper()}")
        for param, value in params.items():
            click.echo(f"  {param}: {value}")

@main.command()
@click.option('--output-dir', default='reports', help='Output directory for reports')
def list_experiments(output_dir):
    """List all completed experiments."""
    try:
        if not os.path.exists(output_dir):
            click.echo(f"Reports directory not found: {output_dir}")
            return
        
        experiments = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check if it's an experiment directory (has index file)
                index_files = [f for f in os.listdir(item_path) if f.startswith('index_') and f.endswith('.html')]
                if index_files:
                    experiments.append((item, item_path, index_files[0]))
        
        if not experiments:
            click.echo("No experiments found")
            return
        
        click.echo(f"Found {len(experiments)} experiments:")
        click.echo("=" * 80)
        
        for exp_name, exp_path, index_file in sorted(experiments):
            click.echo(f"\nExperiment: {exp_name}")
            click.echo(f"  Path: {exp_path}")
            click.echo(f"  Index: {os.path.join(exp_path, index_file)}")
            
            # Try to read experiment config for more details
            config_files = [f for f in os.listdir(os.path.join(exp_path, 'logs')) if f.startswith('experiment_config_') and f.endswith('.json')]
            if config_files:
                try:
                    import json
                    config_path = os.path.join(exp_path, 'logs', config_files[0])
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    click.echo(f"  Pair: {config.get('backtest_config', {}).get('pair', 'Unknown')}")
                    click.echo(f"  Strategy: {config.get('backtest_config', {}).get('strategy', 'Unknown')}")
                    click.echo(f"  Interval: {config.get('backtest_config', {}).get('interval', 'Unknown')}")
                    click.echo(f"  Created: {config.get('created_at', 'Unknown')}")
                except:
                    pass
        
    except Exception as e:
        click.echo(f"Error listing experiments: {e}", err=True)
        raise click.Abort()

@main.command()
@click.option('--pair', required=True, help='Trading pair (e.g., ETHUSDC)')
@click.option('--strategy', type=click.Choice(['ml-bollinger', 'ml-keltner', 'ml-donchian', 'ml-hybrid']), 
              default='ml-bollinger', help='ML strategy type')
@click.option('--interval', type=click.Choice(['1m', '15m', '1h', '4h', '1d']), default='1h', help='Time interval')
@click.option('--data-file', help='Path to CSV data file (optional)')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--model-type', default='random_forest', help='ML model type (random_forest, gradient_boosting, neural_network)')
@click.option('--ml-weight', default=0.7, help='ML weight for hybrid strategies')
@click.option('--rebalance-threshold', default=0.02, help='Rebalancing threshold')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def ml_backtest(pair, strategy, interval, data_file, start_date, end_date, model_type, ml_weight, rebalance_threshold, verbose):
    """Run ML-based Steer Intent backtesting strategies."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        click.echo(f"Starting ML Steer Intent Backtest")
        click.echo(f"Pair: {pair}, Strategy: {strategy}, Interval: {interval}")
        click.echo(f"Model: {model_type}, ML Weight: {ml_weight}")
        
        # Load data
        if data_file:
            click.echo(f"Loading data from file: {data_file}")
            price_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            click.echo("Fetching data from Binance...")
            fetcher = BinanceDataFetcher()
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=30)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            price_data = fetcher.fetch_klines(pair, interval, start_dt, end_dt)
        
        if price_data is None or len(price_data) == 0:
            click.echo(f"No data found for {pair} {interval}", err=True)
            return
        
        click.echo(f"Loaded {len(price_data)} data points")
        
        # Initialize ML components
        click.echo("Initializing ML components...")
        feature_engineer = SteerFeatureEngineer(lookback_periods=50)
        trainer = SteerMLTrainer(feature_engineer, models_dir="models")
        
        # Train ML strategy
        click.echo("Training ML models...")
        steer_ml_strategy = trainer.train_steer_ml_strategy(
            price_data=price_data,
            strategy_type='bollinger',
            intent_model_type=model_type,
            price_model_type=model_type
        )
        
        # Train ML Bollinger strategy
        ml_bollinger_strategy = trainer.train_ml_bollinger_strategy(
            price_data=price_data,
            intent_model_type=model_type,
            n=20,
            k=2.0,
            ml_weight=ml_weight
        )
        
        # Create strategy based on type
        if strategy == 'ml-bollinger':
            ml_strategy = MLBollingerStrategy(
                ml_bollinger_strategy=ml_bollinger_strategy,
                n=20,
                k=2.0,
                rebalance_cooldown_hours=1
            )
        elif strategy == 'ml-keltner':
            ml_strategy = MLKeltnerStrategy(
                intent_model=steer_ml_strategy.intent_model,
                feature_engineer=steer_ml_strategy.feature_engineer,
                n=20,
                m=2.0,
                ml_weight=ml_weight,
                rebalance_cooldown_hours=1
            )
        elif strategy == 'ml-donchian':
            ml_strategy = MLDonchianStrategy(
                intent_model=steer_ml_strategy.intent_model,
                feature_engineer=steer_ml_strategy.feature_engineer,
                n=20,
                ml_weight=ml_weight,
                rebalance_cooldown_hours=1
            )
        elif strategy == 'ml-hybrid':
            ml_strategy = MLHybridStrategy(
                steer_ml_strategy=steer_ml_strategy,
                traditional_weight=1.0 - ml_weight,
                ml_weight=ml_weight,
                rebalance_cooldown_hours=4
            )
        else:
            click.echo(f"Unknown strategy type: {strategy}", err=True)
            return
        
        # Run backtest
        click.echo("Running backtest...")
        backtester = Backtester(
            pair=pair,
            strategy=ml_strategy,
            interval=interval,
            start_date=start_date or price_data.index[0].strftime('%Y-%m-%d'),
            end_date=end_date or price_data.index[-1].strftime('%Y-%m-%d')
        )
        
        results = backtester.run_backtest(price_data)
        
        # Generate reports
        click.echo("Generating reports...")
        backtester.generate_reports("reports")
        
        click.echo("ML backtest completed successfully!")
        
        # Print ML strategy statistics
        ml_stats = ml_strategy.get_strategy_info()
        click.echo(f"\nML Strategy Statistics:")
        click.echo(f"Strategy Type: {ml_stats.get('strategy_type', 'Unknown')}")
        click.echo(f"Rebalance Count: {ml_stats.get('rebalance_count', 0)}")
        if 'avg_ml_confidence' in ml_stats:
            click.echo(f"Avg ML Confidence: {ml_stats['avg_ml_confidence']:.3f}")
        
        # Print evaluation results
        evaluation_results = trainer.evaluate_models()
        click.echo(f"\nModel Evaluation:")
        for model_type, results in evaluation_results.items():
            click.echo(f"{model_type}: {results}")
        
    except Exception as e:
        click.echo(f"ML backtest failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()

@main.command()
@click.option('--pair', required=True, help='Trading pair (e.g., ETHUSDC)')
@click.option('--interval', type=click.Choice(['1m', '15m', '1h', '4h', '1d']), default='1h', help='Time interval')
@click.option('--data-file', help='Path to CSV data file (optional)')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--strategy-type', default='bollinger', help='Strategy type (bollinger, keltner, donchian)')
@click.option('--model-type', default='random_forest', help='ML model type')
@click.option('--ml-weight', default=0.7, help='ML weight for hybrid strategies')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def train_ml_models(pair, interval, data_file, start_date, end_date, strategy_type, model_type, ml_weight, verbose):
    """Train ML models for Steer Intent strategies."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        click.echo(f"Training ML Models for Steer Intent")
        click.echo(f"Pair: {pair}, Interval: {interval}")
        click.echo(f"Strategy: {strategy_type}, Model: {model_type}")
        
        # Load data
        if data_file:
            click.echo(f"Loading data from file: {data_file}")
            price_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            click.echo("Fetching data from Binance...")
            fetcher = BinanceDataFetcher()
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=30)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            price_data = fetcher.fetch_klines(pair, interval, start_dt, end_dt)
        
        if price_data is None or len(price_data) == 0:
            click.echo(f"No data found for {pair} {interval}", err=True)
            return
        
        click.echo(f"Loaded {len(price_data)} data points")
        
        # Initialize ML components
        click.echo("Initializing ML components...")
        feature_engineer = SteerFeatureEngineer(lookback_periods=50)
        trainer = SteerMLTrainer(feature_engineer, models_dir="models")
        
        # Train models
        click.echo("Training ML models...")
        steer_ml_strategy = trainer.train_steer_ml_strategy(
            price_data=price_data,
            strategy_type=strategy_type,
            intent_model_type=model_type,
            price_model_type=model_type
        )
        
        # Save training results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"models/training_results_{pair}_{interval}_{strategy_type}_{timestamp}.joblib"
        trainer.save_training_results(results_file)
        
        click.echo(f"Training completed successfully!")
        click.echo(f"Models saved to: models/")
        click.echo(f"Training results saved to: {results_file}")
        
        # Print evaluation results
        evaluation_results = trainer.evaluate_models()
        click.echo(f"\nModel Evaluation:")
        for model_type, results in evaluation_results.items():
            click.echo(f"{model_type}: {results}")
        
    except Exception as e:
        click.echo(f"ML training failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()

if __name__ == '__main__':
    main()
