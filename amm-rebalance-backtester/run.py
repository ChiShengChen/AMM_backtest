#!/usr/bin/env python3
"""
AMM Dynamic Rebalancing Backtester CLI

Main entry point for running experiments and generating reports.
"""

import click
import yaml
import logging
from pathlib import Path
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
        
        # Generate basic report
        console.print("Generating report...")
        plot_gen = PlotGenerator()
        table_gen = TableGenerator()
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary_df = results['summary']
        summary_df.to_csv('results/quick_test_summary.csv', index=False)
        
        # Generate basic plots
        plot_gen.plot_equity_curves(results, save_path='reports/figs/quick_equity_curves.png')
        plot_gen.plot_apr_mdd_scatter(results, save_path='reports/figs/quick_apr_mdd_scatter.png')
        
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
        
        # Generate comprehensive reports
        console.print("Generating reports...")
        
        # Initialize reporting components
        plot_generator = PlotGenerator(config_data)
        table_generator = TableGenerator()
        strategy_recorder = StrategyRecorder()
        
        # Generate all plots
        plot_generator.plot_equity_curves(results, "reports/figs/equity_curves.png")
        plot_generator.plot_apr_mdd_scatter(results, "reports/figs/apr_mdd_scatter.png")
        plot_generator.plot_fee_vs_price_pnl(results, "reports/figs/fee_vs_price_pnl.png")
        plot_generator.plot_sensitivity_heatmap(results, "reports/figs/sensitivity_heatmap.png")
        plot_generator.plot_gas_frequency_contour(results, "reports/figs/gas_frequency_contour.png")
        plot_generator.plot_il_curve(results, "reports/figs/il_curve.png")
        plot_generator.plot_lvr_estimates(results, "reports/figs/lvr_estimates.png")
        
        # Generate tables and save results
        table_generator.generate_summary_stats(results)
        
        # Save strategy parameters and methodology
        console.print("Recording strategy parameters and methodology...")
        strategy_record = strategy_recorder.record_strategy_parameters(results, best_params)
        strategy_recorder.save_strategy_record(strategy_record)
        strategy_recorder.save_strategy_summary_csv(results)
        strategy_report_path = strategy_recorder.generate_strategy_report(results, best_params)
        
        console.print(f"Strategy report saved to: {strategy_report_path}")
        
        # Save results to CSV
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
        plot_gen.plot_equity_curves(summary_df, save_path='reports/figs/equity_curves.png')
        plot_gen.plot_apr_mdd_scatter(summary_df, save_path='reports/figs/apr_mdd_scatter.png')
        
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

if __name__ == '__main__':
    cli()
