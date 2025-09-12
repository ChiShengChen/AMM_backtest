#!/usr/bin/env python3
"""
Test script for the new report organization system.
"""

import os
import tempfile
import shutil
from datetime import datetime
import pandas as pd

def test_report_organization():
    """Test the new report organization system."""
    print("Testing Report Organization System")
    print("=" * 50)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create mock backtest results
        mock_results = {
            "run_id": "test123",
            "pair": "ETHUSDC",
            "strategy": "bollinger",
            "interval": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "performance": {
                "total_return_pct": 15.5,
                "max_drawdown_pct": -8.2,
                "sharpe_ratio": 1.8,
                "rebalance_count": 45
            },
            "baselines": {
                "hodl_50_50": {
                    "total_return_pct": 12.3,
                    "max_drawdown_pct": -6.1
                },
                "single_asset": {
                    "total_return_pct": 18.7,
                    "max_drawdown_pct": -12.4
                }
            },
            "equity_curves": {
                "strategy": [
                    {"timestamp": "2024-01-01", "total_value": 10000},
                    {"timestamp": "2024-01-31", "total_value": 11550}
                ],
                "hodl_50_50": [
                    {"timestamp": "2024-01-01", "total_value": 10000},
                    {"timestamp": "2024-01-31", "total_value": 11230}
                ],
                "single_asset": [
                    {"timestamp": "2024-01-01", "total_value": 10000},
                    {"timestamp": "2024-01-31", "total_value": 11870}
                ]
            }
        }
        
        try:
            # Import the ReportGenerator
            from steerbt.reports import ReportGenerator
            
            # Create report generator
            generator = ReportGenerator(mock_results, temp_dir)
            
            print(f"✓ Created ReportGenerator")
            print(f"  Experiment name: {generator.experiment_name}")
            print(f"  Experiment directory: {generator.experiment_dir}")
            
            # Test directory structure creation
            expected_dirs = [
                generator.experiment_dir,
                generator.figs_dir,
                generator.data_dir,
                generator.logs_dir
            ]
            
            for dir_path in expected_dirs:
                if os.path.exists(dir_path):
                    print(f"✓ Directory created: {dir_path}")
                else:
                    print(f"✗ Directory missing: {dir_path}")
                    return False
            
            # Test report generation
            print("\nGenerating reports...")
            report_files = generator.generate_all_reports()
            
            print(f"✓ Generated {len(report_files)} report files:")
            for report_type, filepath in report_files.items():
                if filepath and os.path.exists(filepath):
                    print(f"  ✓ {report_type}: {filepath}")
                else:
                    print(f"  ✗ {report_type}: {filepath}")
            
            # Test experiment config generation
            config_file = generator.generate_experiment_config()
            if os.path.exists(config_file):
                print(f"✓ Experiment config: {config_file}")
            else:
                print(f"✗ Experiment config missing: {config_file}")
                return False
            
            # Test index generation
            index_file = generator.generate_experiment_index()
            if os.path.exists(index_file):
                print(f"✓ Experiment index: {index_file}")
            else:
                print(f"✗ Experiment index missing: {index_file}")
                return False
            
            # Verify file structure
            print(f"\nVerifying file structure in {generator.experiment_dir}:")
            for root, dirs, files in os.walk(generator.experiment_dir):
                level = root.replace(generator.experiment_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
            
            print("\n✓ All tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_cleanup_tool():
    """Test the cleanup tool functionality."""
    print("\nTesting Cleanup Tool")
    print("=" * 30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create mock scattered files
        mock_files = [
            "equity_curves_abc123.png",
            "drawdown_curves_abc123.png", 
            "lvr_analysis_abc123.png",
            "equity_curves_abc123.csv",
            "summary_report_abc123.txt",
            "results_abc123.json"
        ]
        
        # Create scattered files
        for filename in mock_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Mock content for {filename}")
            print(f"✓ Created mock file: {filename}")
        
        try:
            # Import the cleanup tool
            from cleanup_reports import ReportCleaner
            
            # Create cleaner
            cleaner = ReportCleaner(temp_dir)
            
            # Test scanning
            scattered_files = cleaner.scan_scattered_files()
            total_files = sum(len(files) for files in scattered_files.values())
            print(f"✓ Scanned {total_files} scattered files")
            
            # Test grouping by run ID
            all_files = []
            for file_list in scattered_files.values():
                all_files.extend(file_list)
            
            grouped_files = cleaner.group_files_by_run_id(all_files)
            print(f"✓ Grouped files by run ID: {list(grouped_files.keys())}")
            
            # Test dry run organization
            stats = cleaner.organize_files(dry_run=True)
            print(f"✓ Dry run organization: {stats}")
            
            print("✓ Cleanup tool tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Cleanup tool test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Running Report Organization Tests")
    print("=" * 50)
    
    # Test report organization
    test1_passed = test_report_organization()
    
    # Test cleanup tool
    test2_passed = test_cleanup_tool()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Report organization system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
