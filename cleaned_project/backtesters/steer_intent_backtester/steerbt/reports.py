"""
Reports generation for CLMM backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any, Optional
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates charts and reports for backtesting results."""
    
    def __init__(self, results: Dict[str, Any], output_dir: str = "reports"):
        self.results = results
        self.run_id = results.get("run_id", "unknown")
        
        # Create experiment-specific directory structure
        self.experiment_name = self._generate_experiment_name()
        self.experiment_dir = os.path.join(output_dir, self.experiment_name)
        
        # Create subdirectories for organized output
        self.figs_dir = os.path.join(self.experiment_dir, "figs")
        self.data_dir = os.path.join(self.experiment_dir, "data")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        
        # Create all directories
        for dir_path in [self.experiment_dir, self.figs_dir, self.data_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        logger.info(f"Created experiment directory: {self.experiment_dir}")
    
    def _generate_experiment_name(self) -> str:
        """Generate a descriptive experiment name based on results."""
        pair = self.results.get("pair", "unknown")
        strategy = self.results.get("strategy", "unknown")
        interval = self.results.get("interval", "unknown")
        
        # Get date range
        start_date = self.results.get("start_date", "")
        end_date = self.results.get("end_date", "")
        
        # Format dates
        if start_date and end_date:
            try:
                start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
                end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
                date_range = f"{start_str}_{end_str}"
            except:
                date_range = f"{start_date}_{end_date}".replace("-", "").replace(" ", "_")
        else:
            date_range = datetime.now().strftime("%Y%m%d")
        
        # Create experiment name
        experiment_name = f"{pair}_{strategy}_{interval}_{date_range}_{self.run_id}"
        
        return experiment_name
    
    def generate_all_reports(self) -> Dict[str, str]:
        """
        Generate all reports and charts.
        
        Returns:
            Dictionary mapping report type to file path
        """
        report_files = {}
        
        try:
            # Generate equity curve chart
            equity_file = self.plot_equity()
            report_files["equity"] = equity_file
            
            # Generate drawdown chart
            drawdown_file = self.plot_drawdown()
            report_files["drawdown"] = drawdown_file
            
            # Generate LVR proxy chart
            lvr_file = self.plot_lvr()
            report_files["lvr"] = lvr_file
            
            # Export CSV data
            csv_file = self.export_csv()
            report_files["csv"] = csv_file
            
            # Generate summary report
            summary_file = self.generate_summary_report()
            report_files["summary"] = summary_file
            
            # Generate experiment configuration
            config_file = self.generate_experiment_config()
            report_files["config"] = config_file
            
            # Generate experiment index
            index_file = self.generate_experiment_index()
            report_files["index"] = index_file
            
            logger.info(f"Generated all reports for run {self.run_id}")
            logger.info(f"Experiment directory: {self.experiment_dir}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise
        
        return report_files
    
    def plot_equity(self) -> str:
        """
        Plot equity curves for strategy and baselines.
        
        Returns:
            Path to saved chart file
        """
        equity_curves = self.results.get("equity_curves", {})
        
        if not equity_curves:
            raise ValueError("No equity curve data found in results")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot strategy equity
        if "strategy" in equity_curves:
            strategy_df = pd.DataFrame(equity_curves["strategy"])
            if not strategy_df.empty:
                strategy_df["timestamp"] = pd.to_datetime(strategy_df["timestamp"])
                ax.plot(strategy_df["timestamp"], strategy_df["total_value"], 
                       label="Strategy", linewidth=2, color='blue')
        
        # Plot HODL 50:50 equity
        if "hodl_50_50" in equity_curves:
            hodl_df = pd.DataFrame(equity_curves["hodl_50_50"])
            if not hodl_df.empty:
                hodl_df["timestamp"] = pd.to_datetime(hodl_df["timestamp"])
                ax.plot(hodl_df["timestamp"], hodl_df["total_value"], 
                       label="HODL 50:50", linewidth=2, color='green', linestyle='--')
        
        # Plot single asset equity
        if "single_asset" in equity_curves:
            single_df = pd.DataFrame(equity_curves["single_asset"])
            if not single_df.empty:
                single_df["timestamp"] = pd.to_datetime(single_df["timestamp"])
                ax.plot(single_df["timestamp"], single_df["total_value"], 
                       label="Single Asset", linewidth=2, color='red', linestyle=':')
        
        # Customize chart
        ax.set_title(f"Equity Curves - {self.results.get('pair', 'Unknown')} - {self.results.get('strategy', 'Unknown')}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value (USD)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance summary
        performance = self.results.get("performance", {})
        if performance:
            summary_text = f"Total Return: {performance.get('total_return_pct', 0):.2f}%\n"
            summary_text += f"Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%\n"
            summary_text += f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}"
            
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save chart
        filename = f"equity_curves_{self.run_id}.png"
        filepath = os.path.join(self.figs_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated equity chart: {filepath}")
        return filepath
    
    def plot_drawdown(self) -> str:
        """
        Plot drawdown curves for strategy and baselines.
        
        Returns:
            Path to saved chart file
        """
        equity_curves = self.results.get("equity_curves", {})
        
        if not equity_curves:
            raise ValueError("No equity curve data found in results")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot strategy drawdown
        if "strategy" in equity_curves:
            strategy_df = pd.DataFrame(equity_curves["strategy"])
            if not strategy_df.empty:
                strategy_df["timestamp"] = pd.to_datetime(strategy_df["timestamp"])
                if "drawdown" in strategy_df.columns:
                    ax.plot(strategy_df["timestamp"], strategy_df["drawdown"], 
                           label="Strategy", linewidth=2, color='blue')
        
        # Plot HODL 50:50 drawdown
        if "hodl_50_50" in equity_curves:
            hodl_df = pd.DataFrame(equity_curves["hodl_50_50"])
            if not hodl_df.empty:
                hodl_df["timestamp"] = pd.to_datetime(hodl_df["timestamp"])
                if "drawdown" in hodl_df.columns:
                    ax.plot(hodl_df["timestamp"], hodl_df["drawdown"], 
                           label="HODL 50:50", linewidth=2, color='green', linestyle='--')
        
        # Plot single asset drawdown
        if "single_asset" in equity_curves:
            single_df = pd.DataFrame(equity_curves["single_asset"])
            if not single_df.empty:
                single_df["timestamp"] = pd.to_datetime(single_df["timestamp"])
                if "drawdown" in single_df.columns:
                    ax.plot(single_df["timestamp"], single_df["drawdown"], 
                           label="Single Asset", linewidth=2, color='red', linestyle=':')
        
        # Customize chart
        ax.set_title(f"Drawdown Curves - {self.results.get('pair', 'Unknown')} - {self.results.get('strategy', 'Unknown')}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add drawdown summary
        performance = self.results.get("performance", {})
        if performance:
            max_dd = performance.get('max_drawdown_pct', 0)
            summary_text = f"Max Drawdown: {max_dd:.2f}%"
            
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Save chart
        filename = f"drawdown_curves_{self.run_id}.png"
        filepath = os.path.join(self.figs_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated drawdown chart: {filepath}")
        return filepath
    
    def plot_lvr(self) -> str:
        """
        Plot LVR proxy curve.
        
        Returns:
            Path to saved chart file
        """
        il_metrics = self.results.get("impermanent_loss", {})
        
        if not il_metrics:
            logger.warning("No impermanent loss data found, skipping LVR chart")
            return ""
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot impermanent loss
        if "impermanent_loss_pct" in il_metrics:
            il_data = il_metrics["impermanent_loss_pct"]
            if il_data:
                il_df = pd.DataFrame(list(il_data.items()), columns=["timestamp", "il_pct"])
                il_df["timestamp"] = pd.to_datetime(il_df["timestamp"])
                
                ax1.plot(il_df["timestamp"], il_df["il_pct"], 
                        label="Impermanent Loss", linewidth=2, color='orange')
                ax1.set_title("Impermanent Loss vs HODL 50:50", fontsize=12, fontweight='bold')
                ax1.set_ylabel("IL (%)", fontsize=10)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Plot LVR proxy
        if "lvr_proxy_pct" in il_metrics:
            lvr_data = il_metrics["lvr_proxy_pct"]
            if lvr_data:
                lvr_df = pd.DataFrame(list(lvr_data.items()), columns=["timestamp", "lvr_pct"])
                lvr_df["timestamp"] = pd.to_datetime(lvr_df["timestamp"])
                
                ax2.plot(lvr_df["timestamp"], lvr_df["lvr_pct"], 
                        label="LVR Proxy", linewidth=2, color='purple')
                ax2.set_title("LVR (Loss-Versus-Rebalancing) Proxy", fontsize=12, fontweight='bold')
                ax2.set_xlabel("Date", fontsize=10)
                ax2.set_ylabel("LVR (%)", fontsize=10)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add summary statistics
        summary_text = f"Avg IL: {il_metrics.get('avg_il', 0):.2f}%\n"
        summary_text += f"Max IL: {il_metrics.get('max_il', 0):.2f}%\n"
        summary_text += f"Avg LVR: {il_metrics.get('avg_lvr', 0):.2f}%\n"
        summary_text += f"Max LVR: {il_metrics.get('max_lvr', 0):.2f}%"
        
        fig.suptitle(f"Impermanent Loss & LVR Analysis - {self.results.get('pair', 'Unknown')} - {self.results.get('strategy', 'Unknown')}", 
                    fontsize=14, fontweight='bold')
        
        # Add summary text
        fig.text(0.02, 0.98, summary_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save chart
        filename = f"lvr_analysis_{self.run_id}.png"
        filepath = os.path.join(self.figs_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated LVR chart: {filepath}")
        return filepath
    
    def export_csv(self) -> str:
        """
        Export equity curves and metrics to CSV.
        
        Returns:
            Path to saved CSV file
        """
        equity_curves = self.results.get("equity_curves", {})
        
        if not equity_curves:
            raise ValueError("No equity curve data found in results")
        
        # Combine all equity curves
        all_data = []
        
        for strategy_name, curve_data in equity_curves.items():
            if curve_data:
                df = pd.DataFrame(curve_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["strategy"] = strategy_name
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No valid equity curve data found")
        
        # Combine and sort
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(["strategy", "timestamp"])
        
        # Add performance metrics
        performance = self.results.get("performance", {})
        if performance:
            # Add strategy performance as metadata
            metadata = pd.DataFrame([{
                "strategy": "strategy_metadata",
                "timestamp": combined_df["timestamp"].min(),
                "total_return_pct": performance.get("total_return_pct", 0),
                "max_drawdown_pct": performance.get("max_drawdown_pct", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "rebalance_count": performance.get("rebalance_count", 0),
                "total_fees_paid": performance.get("total_fees_paid", 0)
            }])
            combined_df = pd.concat([metadata, combined_df], ignore_index=True)
        
        # Save to CSV
        filename = f"equity_curves_{self.run_id}.csv"
        filepath = os.path.join(self.data_dir, filename)
        combined_df.to_csv(filepath, index=False)
        
        logger.info(f"Exported CSV data: {filepath}")
        return filepath
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Path to saved report file
        """
        filename = f"summary_report_{self.run_id}.txt"
        filepath = os.path.join(self.logs_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BACKTEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Pair: {self.results.get('pair', 'Unknown')}\n")
            f.write(f"Interval: {self.results.get('interval', 'Unknown')}\n")
            f.write(f"Strategy: {self.results.get('strategy', 'Unknown')}\n")
            f.write(f"Period: {self.results.get('start_date', 'Unknown')} to {self.results.get('end_date', 'Unknown')}\n")
            f.write(f"Total Bars: {self.results.get('total_bars', 0)}\n\n")
            
            # Strategy performance
            performance = self.results.get("performance", {})
            if performance:
                f.write("STRATEGY PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Return: {performance.get('total_return_pct', 0):.2f}%\n")
                f.write(f"Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%\n")
                f.write(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Rebalance Count: {performance.get('rebalance_count', 0)}\n")
                f.write(f"Total Fees Paid: ${performance.get('total_fees_paid', 0):.2f}\n\n")
            
            # Baseline performance
            baselines = self.results.get("baselines", {})
            if baselines:
                f.write("BASELINE PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                
                if "hodl_50_50" in baselines:
                    hodl = baselines["hodl_50_50"]
                    f.write(f"HODL 50:50 Return: {hodl.get('total_return_pct', 0):.2f}%\n")
                    f.write(f"HODL 50:50 Max DD: {hodl.get('max_drawdown_pct', 0):.2f}%\n")
                    f.write(f"HODL 50:50 Sharpe: {hodl.get('sharpe_ratio', 0):.2f}\n\n")
                
                if "single_asset" in baselines:
                    single = baselines["single_asset"]
                    f.write(f"Single Asset Return: {single.get('total_return_pct', 0):.2f}%\n")
                    f.write(f"Single Asset Max DD: {single.get('max_drawdown_pct', 0):.2f}%\n")
                    f.write(f"Single Asset Sharpe: {single.get('sharpe_ratio', 0):.2f}\n\n")
            
            # Impermanent loss metrics
            il_metrics = self.results.get("impermanent_loss", {})
            if il_metrics:
                f.write("IMPERMANENT LOSS & LVR:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average IL: {il_metrics.get('avg_il', 0):.2f}%\n")
                f.write(f"Maximum IL: {il_metrics.get('max_il', 0):.2f}%\n")
                f.write(f"Average LVR: {il_metrics.get('avg_lvr', 0):.2f}%\n")
                f.write(f"Maximum LVR: {il_metrics.get('max_lvr', 0):.2f}%\n\n")
            
            # Strategy information
            strategy_info = self.results.get("strategy_info", {})
            if strategy_info:
                f.write("STRATEGY DETAILS:\n")
                f.write("-" * 40 + "\n")
                for key, value in strategy_info.items():
                    if key not in ["parameters"]:  # Skip complex parameters
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Generated summary report: {filepath}")
        return filepath
    
    def generate_experiment_config(self) -> str:
        """
        Generate experiment configuration file.
        
        Returns:
            Path to saved config file
        """
        import json
        
        filename = f"experiment_config_{self.run_id}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        # Extract configuration from results
        config = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "backtest_config": self.results.get("config", {}),
            "strategy_info": self.results.get("strategy_info", {}),
            "performance_summary": self.results.get("performance", {}),
            "baseline_performance": self.results.get("baselines", {}),
            "file_structure": {
                "figs_dir": "figs/",
                "data_dir": "data/",
                "logs_dir": "logs/"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Generated experiment config: {filepath}")
        return filepath
    
    def generate_experiment_index(self) -> str:
        """
        Generate experiment index HTML file for easy browsing.
        
        Returns:
            Path to saved index file
        """
        filename = f"index_{self.run_id}.html"
        filepath = os.path.join(self.experiment_dir, filename)
        
        # Get performance metrics
        performance = self.results.get("performance", {})
        baselines = self.results.get("baselines", {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Results - {self.experiment_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007acc; }}
        .metric-label {{ font-weight: bold; color: #333; }}
        .metric-value {{ color: #007acc; font-size: 1.2em; }}
        .file-list {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .file-item {{ margin: 10px 0; padding: 10px; background-color: white; border-radius: 3px; border: 1px solid #ddd; }}
        .file-item a {{ text-decoration: none; color: #007acc; }}
        .file-item a:hover {{ text-decoration: underline; }}
        .performance-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .performance-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }}
        .performance-card h3 {{ margin-top: 0; color: #333; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Results: {self.experiment_name}</h1>
        
        <div class="metric">
            <div class="metric-label">Run ID:</div>
            <div class="metric-value">{self.run_id}</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Generated:</div>
            <div class="metric-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <h2>Performance Summary</h2>
        <div class="performance-grid">
            <div class="performance-card">
                <h3>Strategy Performance</h3>
                <div class="metric">
                    <div class="metric-label">Total Return:</div>
                    <div class="metric-value {'positive' if performance.get('total_return_pct', 0) > 0 else 'negative'}">{performance.get('total_return_pct', 0):.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown:</div>
                    <div class="metric-value negative">{performance.get('max_drawdown_pct', 0):.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio:</div>
                    <div class="metric-value {'positive' if performance.get('sharpe_ratio', 0) > 0 else 'negative'}">{performance.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Rebalance Count:</div>
                    <div class="metric-value neutral">{performance.get('rebalance_count', 0)}</div>
                </div>
            </div>
            
            <div class="performance-card">
                <h3>Baseline Comparison</h3>
                <div class="metric">
                    <div class="metric-label">HODL 50:50 Return:</div>
                    <div class="metric-value {'positive' if baselines.get('hodl_50_50', {}).get('total_return_pct', 0) > 0 else 'negative'}">{baselines.get('hodl_50_50', {}).get('total_return_pct', 0):.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Single Asset Return:</div>
                    <div class="metric-value {'positive' if baselines.get('single_asset', {}).get('total_return_pct', 0) > 0 else 'negative'}">{baselines.get('single_asset', {}).get('total_return_pct', 0):.2f}%</div>
                </div>
            </div>
        </div>
        
        <h2>Generated Files</h2>
        <div class="file-list">
            <h3>Charts (figs/)</h3>
            <div class="file-item">
                <a href="figs/equity_curves_{self.run_id}.png">Equity Curves Chart</a>
            </div>
            <div class="file-item">
                <a href="figs/drawdown_curves_{self.run_id}.png">Drawdown Curves Chart</a>
            </div>
            <div class="file-item">
                <a href="figs/lvr_analysis_{self.run_id}.png">LVR Analysis Chart</a>
            </div>
            
            <h3>Data (data/)</h3>
            <div class="file-item">
                <a href="data/equity_curves_{self.run_id}.csv">Equity Curves Data (CSV)</a>
            </div>
            
            <h3>Reports (logs/)</h3>
            <div class="file-item">
                <a href="logs/summary_report_{self.run_id}.txt">Summary Report (TXT)</a>
            </div>
            <div class="file-item">
                <a href="logs/experiment_config_{self.run_id}.json">Experiment Configuration (JSON)</a>
            </div>
        </div>
        
        <h2>Experiment Details</h2>
        <div class="metric">
            <div class="metric-label">Pair:</div>
            <div class="metric-value">{self.results.get('pair', 'Unknown')}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Strategy:</div>
            <div class="metric-value">{self.results.get('strategy', 'Unknown')}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Interval:</div>
            <div class="metric-value">{self.results.get('interval', 'Unknown')}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Period:</div>
            <div class="metric-value">{self.results.get('start_date', 'Unknown')} to {self.results.get('end_date', 'Unknown')}</div>
        </div>
    </div>
</body>
</html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated experiment index: {filepath}")
        return filepath
