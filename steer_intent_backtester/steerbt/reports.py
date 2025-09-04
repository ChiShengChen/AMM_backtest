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
        self.output_dir = output_dir
        self.run_id = results.get("run_id", "unknown")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
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
            
            logger.info(f"Generated all reports for run {self.run_id}")
            
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
        filepath = os.path.join(self.output_dir, filename)
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
        filepath = os.path.join(self.output_dir, filename)
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
        filepath = os.path.join(self.output_dir, filename)
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
        filepath = os.path.join(self.output_dir, filename)
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
        filepath = os.path.join(self.output_dir, filename)
        
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
