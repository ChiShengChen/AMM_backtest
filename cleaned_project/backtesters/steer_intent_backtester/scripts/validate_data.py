#!/usr/bin/env python3
"""
Data validation script for Steer Intent Backtester.
This script demonstrates the data validation functionality mentioned in README.
"""

import os
import sys
import pandas as pd
import glob
from datetime import datetime, timedelta
import argparse

def validate_data_file(filepath):
    """Validate a single data file and return validation results."""
    try:
        df = pd.read_csv(filepath)
        
        # Basic file info
        file_size = os.path.getsize(filepath) / 1024  # KB
        record_count = len(df)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Data quality checks
        missing_values = df.isnull().sum().sum()
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        
        # Price and volume checks
        price_range = (df['low'].min(), df['high'].max())
        volume_total = df['volume'].sum()
        
        # Time range
        time_range = (df['timestamp'].min(), df['timestamp'].max())
        time_span = time_range[1] - time_range[0]
        
        # Data consistency checks
        price_consistency = all(df['low'] <= df['high']) and all(df['low'] <= df['close']) and all(df['close'] <= df['high'])
        volume_consistency = all(df['volume'] >= 0)
        
        # Calculate expected records based on interval
        filename = os.path.basename(filepath)
        if '1h' in filename:
            expected_interval = timedelta(hours=1)
        elif '1d' in filename:
            expected_interval = timedelta(days=1)
        elif '4h' in filename:
            expected_interval = timedelta(hours=4)
        elif '15m' in filename:
            expected_interval = timedelta(minutes=15)
        else:
            expected_interval = None
        
        if expected_interval and record_count > 1:
            time_gaps = df['timestamp'].diff().dropna()
            expected_gaps = time_gaps.value_counts()
            most_common_gap = expected_gaps.index[0] if len(expected_gaps) > 0 else None
            gap_consistency = most_common_gap == expected_interval if most_common_gap else False
        else:
            gap_consistency = True
        
        return {
            'filepath': filepath,
            'filename': filename,
            'file_size_kb': file_size,
            'record_count': record_count,
            'missing_values': missing_values,
            'duplicate_timestamps': duplicate_timestamps,
            'price_range': price_range,
            'volume_total': volume_total,
            'time_range': time_range,
            'time_span': time_span,
            'price_consistency': price_consistency,
            'volume_consistency': volume_consistency,
            'gap_consistency': gap_consistency,
            'validation_passed': (
                missing_values == 0 and 
                duplicate_timestamps == 0 and 
                price_consistency and 
                volume_consistency and
                gap_consistency
            )
        }
        
    except Exception as e:
        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'error': str(e),
            'validation_passed': False
        }

def print_validation_report(results):
    """Print a formatted validation report."""
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    total_files = len(results)
    passed_files = sum(1 for r in results if r.get('validation_passed', False))
    failed_files = total_files - passed_files
    
    print(f"Total files: {total_files}")
    print(f"Validation passed: {passed_files}")
    print(f"Validation failed: {failed_files}")
    print(f"Success rate: {passed_files/total_files*100:.1f}%")
    
    if failed_files > 0:
        print(f"\n❌ VALIDATION FAILURES:")
        for result in results:
            if not result.get('validation_passed', False):
                if 'error' in result:
                    print(f"  {result['filename']}: ERROR - {result['error']}")
                else:
                    issues = []
                    if result.get('missing_values', 0) > 0:
                        issues.append(f"Missing values: {result['missing_values']}")
                    if result.get('duplicate_timestamps', 0) > 0:
                        issues.append(f"Duplicate timestamps: {result['duplicate_timestamps']}")
                    if not result.get('price_consistency', True):
                        issues.append("Price consistency issues")
                    if not result.get('volume_consistency', True):
                        issues.append("Volume consistency issues")
                    if not result.get('gap_consistency', True):
                        issues.append("Time gap consistency issues")
                    
                    print(f"  {result['filename']}: {'; '.join(issues)}")
    
    print(f"\n✅ VALIDATION SUCCESSES:")
    for result in results:
        if result.get('validation_passed', False):
            print(f"  {result['filename']}: {result['record_count']} records, "
                  f"{result['file_size_kb']:.1f} KB, "
                  f"{result['time_range'][0].date()} to {result['time_range'][1].date()}")

def main():
    parser = argparse.ArgumentParser(description='Validate data files for Steer Intent Backtester')
    parser.add_argument('--data-dir', default='data', help='Data directory to validate')
    parser.add_argument('--pattern', default='*.csv', help='File pattern to match')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find data files
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    csv_files = glob.glob(os.path.join(data_dir, args.pattern))
    
    if not csv_files:
        print(f"No CSV files found in '{data_dir}' matching pattern '{args.pattern}'")
        sys.exit(0)
    
    print(f"Found {len(csv_files)} CSV files to validate...")
    
    # Validate each file
    results = []
    for filepath in csv_files:
        if args.verbose:
            print(f"Validating: {filepath}")
        
        result = validate_data_file(filepath)
        results.append(result)
    
    # Print report
    print_validation_report(results)
    
    # Exit with error code if any validation failed
    if any(not r.get('validation_passed', False) for r in results):
        sys.exit(1)

if __name__ == '__main__':
    main()
