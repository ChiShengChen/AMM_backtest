#!/usr/bin/env python3
"""
AMM Dynamic Rebalancing Backtester CLI

Main entry point for running experiments and generating reports.
"""

import click
import yaml
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sys
import os
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.io.loader import DataLoader, ValidationConfig
from src.core.engine import BacktestEngine
from src.opt.search import OptunaOptimizer
from src.reporting.plots import PlotGenerator
from src.reporting.tables import TableGenerator
from src.reporting.strategy_recorder import StrategyRecorder
from src.ml import FeatureEngineer, MLTrainer, MLStrategy, RebalancePredictor, VolatilityPredictor
from src.strategies import MLBasedStrategy, MLVolatilityStrategy, MLHybridStrategy

console = Console()

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('amm_backtest.log')
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

def create_directories():
    """Create necessary directories."""
    dirs = ['results', 'reports', 'reports/figs']
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """AMM Dynamic Rebalancing Backtester"""
    setup_logging(verbose)
    create_directories()
    ctx.ensure_object(dict)

@cli.command()
@click.option('--pool', required=True, help='Pool name (e.g., ETHUSDC)')
@click.option('--freq', default='1h', help='Data frequency (1h, 1d)')
@click.option('--fee-mode', type=click.Choice(['exact', 'proxy']), default='proxy', 
              help='Fee calculation mode')
@click.option('--config', default='configs/experiment_default.yaml', 
              help='Configuration file path')
def quick(pool, freq, fee_mode, config):
    """Quick test with recent data (last 60 days)"""
    console.print(f"[green]Running quick test for {pool} with {freq} data[/green]")
    
    # Load config
    config_data = load_config(config)
    config_data['pool'] = pool
    config_data['frequency'] = freq
    config_data['fee_mode'] = fee_mode
    
    # Override for quick test
    config_data['wfa']['train_days'] = 30
    config_data['wfa']['valid_days'] = 15
    config_data['wfa']['test_days'] = 15
    config_data['wfa']['n_trials'] = 10
    
    try:
        # Initialize data loader
        validation_config = ValidationConfig()
        data_loader = DataLoader('data', validation_config)
        
        # Load data
        console.print("Loading data...")
        price_data, pool_data = data_loader.load_pool_data(pool, freq)
        
        if pool_data is None and fee_mode == 'exact':
            console.print("[yellow]Warning: No pool data available, switching to proxy mode[/yellow]")
            config_data['fee_mode'] = 'proxy'
        
        # Initialize backtest engine
        engine = BacktestEngine(config_data)
        
        # Run quick test
        console.print("Running backtest...")
        results = engine.run_quick_test(price_data, pool_data)
        
        # Generate organized reports
        console.print("Generating organized reports...")
        report_files = engine.generate_organized_reports("reports")
        
        console.print(f"Reports generated successfully!")
        console.print(f"Experiment directory: {report_files.get('index', '').replace('/index_', '/').replace('_', '/')}")
        console.print(f"Open index file to view all results: {report_files.get('index', '')}")
        
        # Save summary for backward compatibility
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        summary_df = results['summary']
        summary_df.to_csv('results/quick_test_summary.csv', index=False)
        
        # Display summary table
        table = Table(title=f"Quick Test Results - {pool}")
        table.add_column("Strategy", style="cyan")
        table.add_column("APR (%)", style="green")
        table.add_column("MDD (%)", style="red")
        table.add_column("Sharpe", style="blue")
        table.add_column("Rebalances", style="yellow")
        
        for _, row in summary_df.iterrows():
            table.add_row(
                row['strategy'],
                f"{row['apr']:.2f}",
                f"{row['mdd']:.2f}",
                f"{row['sharpe']:.2f}",
                str(row['rebalance_count'])
            )
        
        console.print(table)
        console.print(f"[green]Quick test completed! Results saved to results/ and reports/figs/[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during quick test: {e}[/red]")
        logging.error(f"Quick test error: {e}", exc_info=True)
        sys.exit(1)

@cli.command()
@click.option('--pool', required=True, help='Pool name (e.g., ETHUSDC)')
@click.option('--freq', default='1h', help='Data frequency (1h, 1d)')
@click.option('--fee-mode', type=click.Choice(['exact', 'proxy']), default='proxy', 
              help='Fee calculation mode')
@click.option('--study-name', required=True, help='Optuna study name')
@click.option('--n-trials', default=50, help='Number of optimization trials')
@click.option('--config', default='configs/experiment_default.yaml', 
              help='Configuration file path')
def full(pool, freq, fee_mode, study_name, n_trials, config):
    """Full Walk-Forward analysis with hyperparameter optimization"""
    console.print(f"[green]Running full analysis for {pool} with {freq} data[/green]")
    console.print(f"Study: {study_name}, Trials: {n_trials}")
    
    # Load config
    config_data = load_config(config)
    config_data['pool'] = pool
    config_data['frequency'] = freq
    config_data['fee_mode'] = fee_mode
    config_data['wfa']['n_trials'] = n_trials
    
    try:
        # Initialize data loader
        validation_config = ValidationConfig()
        data_loader = DataLoader('data', validation_config)
        
        # Load data
        console.print("Loading data...")
        price_data, pool_data = data_loader.load_pool_data(pool, freq)
        
        if pool_data is None and fee_mode == 'exact':
            console.print("[yellow]Warning: No pool data available, switching to proxy mode[/yellow]")
            config_data['fee_mode'] = 'proxy'
        
        # Initialize optimizer
        optimizer = OptunaOptimizer(config_data, study_name)
        
        # Run optimization
        console.print("Running hyperparameter optimization...")
        study = optimizer.optimize(price_data, pool_data)
        
        # Run final evaluation with best parameters
        console.print("Running final evaluation...")
        engine = BacktestEngine(config_data)
        
        try:
            # Get best parameters safely
            if hasattr(study, 'best_params') and study.best_params:
                best_params = study.best_params
                console.print(f"Using best parameters: {best_params}")
            else:
                console.print("[yellow]Warning: No best parameters found, using defaults[/yellow]")
                best_params = None
            
            results = engine.run_full_evaluation(price_data, pool_data, best_params)
            
        except Exception as e:
            console.print(f"[red]Error during final evaluation: {e}[/red]")
            console.print("[yellow]Continuing with basic results...[/yellow]")
            # Try to run with default parameters
            results = engine.run_full_evaluation(price_data, pool_data, None)
        
        # Generate organized reports
        console.print("Generating organized reports...")
        report_files = engine.generate_organized_reports("reports")
        
        console.print(f"Reports generated successfully!")
        console.print(f"Experiment directory: {report_files.get('index', '').replace('/index_', '/').replace('_', '/')}")
        console.print(f"Open index file to view all results: {report_files.get('index', '')}")
        
        # Save strategy parameters and methodology for backward compatibility
        console.print("Recording strategy parameters and methodology...")
        strategy_recorder = StrategyRecorder()
        strategy_record = strategy_recorder.record_strategy_parameters(results, best_params)
        strategy_recorder.save_strategy_record(strategy_record)
        strategy_recorder.save_strategy_summary_csv(results)
        strategy_report_path = strategy_recorder.generate_strategy_report(results, best_params)
        
        console.print(f"Strategy report saved to: {strategy_report_path}")
        
        # Save results to CSV for backward compatibility
        results['summary'].to_csv("results/full_analysis_summary.csv", index=False)
        results['trades'].to_csv("results/full_analysis_trades.csv", index=False)
        
        # Display summary
        table = Table(title=f"Full Analysis Results - {pool}")
        table.add_column("Strategy", style="cyan")
        table.add_column("APR (%)", style="green")
        table.add_column("MDD (%)", style="red")
        table.add_column("Sharpe", style="blue")
        table.add_column("Calmar", style="magenta")
        table.add_column("Rebalances", style="yellow")
        
        for _, row in results['summary'].iterrows():
            table.add_row(
                row['strategy'],
                f"{row['apr']:.2f}",
                f"{row['mdd']:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['calmar']:.2f}",
                str(row['rebalance_count'])
            )
        
        console.print(table)
        console.print(f"[green]Full analysis completed! Results saved to results/ and reports/figs/[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during full analysis: {e}[/red]")
        logging.error(f"Full analysis error: {e}", exc_info=True)
        sys.exit(1)

@cli.command()
@click.option('--pool', required=True, help='Pool name (e.g., ETHUSDC)')
@click.option('--results-dir', default='results', help='Results directory path')
def report(pool, results_dir):
    """Generate reports from existing results"""
    console.print(f"[green]Generating reports for {pool} from {results_dir}[/green]")
    
    try:
        # Check if results exist
        results_path = Path(results_dir)
        if not results_path.exists():
            console.print(f"[red]Results directory not found: {results_dir}[/red]")
            sys.exit(1)
        
        # Load results
        summary_file = results_path / f"{pool.lower()}_summary.csv"
        if not summary_file.exists():
            console.print(f"[red]Summary file not found: {summary_file}[/red]")
            sys.exit(1)
        
        # Load and process results
        summary_df = pd.read_csv(summary_file)
        
        # Generate plots
        console.print("Generating plots...")
        plot_gen = PlotGenerator()
        
        # Generate all plots (you'll need to implement these methods)
        # Generate summary plot with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pool = config_data.get('pool', 'UNKNOWN')
        pool_dir = f"reports/figs/{pool.lower()}"
        os.makedirs(pool_dir, exist_ok=True)
        plot_gen.plot_equity_curves(summary_df, save_path=f'{pool_dir}/{pool}_summary_equity_curves_{timestamp}.png')
        plot_gen.plot_apr_mdd_scatter(summary_df, save_path=f'{pool_dir}/{pool}_summary_apr_mdd_scatter_{timestamp}.png')
        
        console.print(f"[green]Reports generated! Check reports/figs/ directory[/green]")
        
    except Exception as e:
        console.print(f"[red]Error generating reports: {e}[/red]")
        logging.error(f"Report generation error: {e}", exc_info=True)
        sys.exit(1)

@cli.command()
def list_pools():
    """List available pools"""
    try:
        validation_config = ValidationConfig()
        data_loader = DataLoader('data', validation_config)
        
        pools = data_loader.get_available_pools()
        
        if not pools:
            console.print("[yellow]No pools found in data/ directory[/yellow]")
            return
        
        table = Table(title="Available Pools")
        table.add_column("Pool Name", style="cyan")
        table.add_column("Available Frequencies", style="green")
        
        for pool in pools:
            frequencies = data_loader.get_available_frequencies(pool)
            freq_str = ", ".join(frequencies) if frequencies else "None"
            table.add_row(pool, freq_str)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing pools: {e}[/red]")
        logging.error(f"List pools error: {e}", exc_info=True)

@cli.command()
@click.option('--reports-dir', default='reports', help='Reports directory path')
def list_experiments(reports_dir):
    """List all completed experiments"""
    try:
        reports_path = Path(reports_dir)
        if not reports_path.exists():
            console.print(f"[yellow]Reports directory not found: {reports_dir}[/yellow]")
            return
        
        experiments = []
        for item in reports_path.iterdir():
            if item.is_dir():
                # Check if it's an experiment directory (has index file)
                index_files = list(item.glob('index_*.html'))
                if index_files:
                    experiments.append((item.name, item, index_files[0]))
        
        if not experiments:
            console.print("[yellow]No experiments found[/yellow]")
            return
        
        table = Table(title=f"Found {len(experiments)} Experiments")
        table.add_column("Experiment Name", style="cyan")
        table.add_column("Index File", style="green")
        table.add_column("Created", style="yellow")
        
        for exp_name, exp_path, index_file in sorted(experiments):
            # Try to read experiment config for creation date
            config_files = list((exp_path / 'logs').glob('experiment_config_*.json'))
            created_date = "Unknown"
            if config_files:
                try:
                    with open(config_files[0], 'r') as f:
                        config = json.load(f)
                    created_date = config.get('created_at', 'Unknown')[:10]  # Just the date part
                except:
                    pass
            
            table.add_row(exp_name, str(index_file.name), created_date)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing experiments: {e}[/red]")
        logging.error(f"List experiments error: {e}", exc_info=True)

@cli.command()
@click.option('--pool', default='ETHUSDC', help='Trading pair (e.g., ETHUSDC, BTCUSDC)')
@click.option('--freq', default='1d', help='Data frequency (1d, 1h, 1m)')
@click.option('--strategy-type', default='ml-based', help='ML strategy type (ml-based, ml-volatility, ml-hybrid)')
@click.option('--model-type', default='random_forest', help='ML model type (random_forest, gradient_boosting, neural_network)')
@click.option('--rebalance-threshold', default=0.02, help='Rebalancing threshold')
@click.option('--ml-weight', default=0.7, help='ML weight for hybrid strategies')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def ml_backtest(pool, freq, strategy_type, model_type, rebalance_threshold, ml_weight, verbose):
    """Run ML-based backtesting strategies."""
    setup_logging(verbose)
    
    try:
        console.print(f"[bold blue]Starting ML Backtest[/bold blue]")
        console.print(f"Pool: {pool}, Frequency: {freq}")
        console.print(f"Strategy: {strategy_type}, Model: {model_type}")
        
        # Load data
        console.print("Loading data...")
        data_loader = DataLoader()
        price_data = data_loader.load_data(pool, freq)
        
        if price_data is None or len(price_data) == 0:
            console.print(f"[red]No data found for {pool} {freq}[/red]")
            return
        
        console.print(f"Loaded {len(price_data)} data points")
        
        # Initialize ML components
        console.print("Initializing ML components...")
        feature_engineer = FeatureEngineer(lookback_periods=50)
        trainer = MLTrainer(feature_engineer, models_dir="models")
        
        # Train ML strategy
        console.print("Training ML models...")
        ml_strategy = trainer.train_ml_strategy(
            price_data=price_data,
            rebalance_model_type=model_type,
            volatility_model_type=model_type,
            rebalance_threshold=rebalance_threshold
        )
        
        # Create strategy based on type
        if strategy_type == 'ml-based':
            strategy = MLBasedStrategy(
                ml_strategy=ml_strategy,
                initial_width=0.1,
                rebalance_cooldown_hours=1
            )
        elif strategy_type == 'ml-volatility':
            strategy = MLVolatilityStrategy(
                volatility_model=ml_strategy.volatility_model,
                feature_engineer=ml_strategy.feature_engineer,
                base_k_width=1.5,
                rebalance_cooldown_hours=6
            )
        elif strategy_type == 'ml-hybrid':
            strategy = MLHybridStrategy(
                ml_strategy=ml_strategy,
                traditional_weight=1.0 - ml_weight,
                ml_weight=ml_weight,
                rebalance_cooldown_hours=4
            )
        else:
            console.print(f"[red]Unknown strategy type: {strategy_type}[/red]")
            return
        
        # Run backtest
        console.print("Running backtest...")
        engine = BacktestEngine(
            pool=pool,
            frequency=freq,
            strategy=strategy,
            fee_mode='proxy'
        )
        
        results = engine.run_backtest(price_data)
        
        # Generate organized reports
        console.print("Generating organized reports...")
        report_files = engine.generate_organized_reports("reports")
        
        console.print(f"Reports generated successfully!")
        console.print(f"Experiment directory: {report_files.get('index', '').replace('/index_', '/').replace('_', '/')}")
        console.print(f"Open index file to view all results: {report_files.get('index', '')}")
        
        # Print ML strategy statistics
        ml_stats = strategy.get_strategy_info()
        console.print(f"\n[bold green]ML Strategy Statistics:[/bold green]")
        console.print(f"Strategy Type: {ml_stats.get('strategy_type', 'Unknown')}")
        console.print(f"Rebalance Count: {ml_stats.get('rebalance_count', 0)}")
        if 'ml_stats' in ml_stats:
            console.print(f"ML Stats: {ml_stats['ml_stats']}")
        
        # Print evaluation results
        evaluation_results = trainer.evaluate_models()
        console.print(f"\n[bold green]Model Evaluation:[/bold green]")
        for model_type, results in evaluation_results.items():
            console.print(f"{model_type}: {results}")
        
    except Exception as e:
        console.print(f"[red]ML backtest failed: {e}[/red]")
        logging.error(f"ML backtest error: {e}", exc_info=True)

@cli.command()
@click.option('--pool', default='ETHUSDC', help='Trading pair (e.g., ETHUSDC, BTCUSDC)')
@click.option('--freq', default='1d', help='Data frequency (1d, 1h, 1m)')
@click.option('--strategy-type', default='bollinger', help='Strategy type (bollinger, keltner, donchian)')
@click.option('--model-type', default='random_forest', help='ML model type')
@click.option('--ml-weight', default=0.7, help='ML weight for hybrid strategies')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def train_ml_models(pool, freq, strategy_type, model_type, ml_weight, verbose):
    """Train ML models for Steer Intent strategies."""
    setup_logging(verbose)
    
    try:
        console.print(f"[bold blue]Training ML Models[/bold blue]")
        console.print(f"Pool: {pool}, Frequency: {freq}")
        console.print(f"Strategy: {strategy_type}, Model: {model_type}")
        
        # Load data
        console.print("Loading data...")
        data_loader = DataLoader()
        price_data = data_loader.load_data(pool, freq)
        
        if price_data is None or len(price_data) == 0:
            console.print(f"[red]No data found for {pool} {freq}[/red]")
            return
        
        console.print(f"Loaded {len(price_data)} data points")
        
        # Initialize ML components
        console.print("Initializing ML components...")
        feature_engineer = FeatureEngineer(lookback_periods=50)
        trainer = MLTrainer(feature_engineer, models_dir="models")
        
        # Train models
        console.print("Training ML models...")
        ml_strategy = trainer.train_ml_strategy(
            price_data=price_data,
            rebalance_model_type=model_type,
            volatility_model_type=model_type
        )
        
        # Save training results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"models/training_results_{pool}_{freq}_{strategy_type}_{timestamp}.joblib"
        trainer.save_training_results(results_file)
        
        console.print(f"Training completed successfully!")
        console.print(f"Models saved to: models/")
        console.print(f"Training results saved to: {results_file}")
        
        # Print evaluation results
        evaluation_results = trainer.evaluate_models()
        console.print(f"\n[bold green]Model Evaluation:[/bold green]")
        for model_type, results in evaluation_results.items():
            console.print(f"{model_type}: {results}")
        
    except Exception as e:
        console.print(f"[red]ML training failed: {e}[/red]")
        logging.error(f"ML training error: {e}", exc_info=True)

if __name__ == '__main__':
    cli()
