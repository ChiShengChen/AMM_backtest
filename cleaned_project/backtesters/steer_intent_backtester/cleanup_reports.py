#!/usr/bin/env python3
"""
Report cleanup and organization tool for Steer Intent Backtester.

This tool helps organize scattered report files into the new structured format.
"""

import os
import shutil
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple
import click
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportCleaner:
    """Tool for cleaning up and organizing scattered report files."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        self.backup_dir = os.path.join(reports_dir, "backup", datetime.now().strftime("%Y%m%d_%H%M%S"))
        
    def scan_scattered_files(self) -> Dict[str, List[str]]:
        """Scan for scattered report files in the reports directory."""
        scattered_files = {
            "png_files": [],
            "csv_files": [],
            "txt_files": [],
            "json_files": []
        }
        
        if not os.path.exists(self.reports_dir):
            logger.warning(f"Reports directory not found: {self.reports_dir}")
            return scattered_files
        
        for root, dirs, files in os.walk(self.reports_dir):
            # Skip subdirectories that are already organized
            if any(subdir in root for subdir in ['figs', 'data', 'logs']):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.png'):
                    scattered_files["png_files"].append(file_path)
                elif file.endswith('.csv'):
                    scattered_files["csv_files"].append(file_path)
                elif file.endswith('.txt'):
                    scattered_files["txt_files"].append(file_path)
                elif file.endswith('.json'):
                    scattered_files["json_files"].append(file_path)
        
        return scattered_files
    
    def extract_run_id_from_filename(self, filename: str) -> str:
        """Extract run ID from filename."""
        # Common patterns for run IDs in filenames
        patterns = [
            r'_([a-f0-9]{8})\.',  # 8-character hex ID
            r'_([a-f0-9]{6})\.',  # 6-character hex ID
            r'_(\d{8}_\d{6})\.',  # timestamp format
            r'_([a-zA-Z0-9]{6,8})\.'  # general alphanumeric
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def group_files_by_run_id(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by their run ID."""
        grouped = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            run_id = self.extract_run_id_from_filename(filename)
            
            if run_id not in grouped:
                grouped[run_id] = []
            grouped[run_id].append(file_path)
        
        return grouped
    
    def create_backup(self, files: List[str]) -> bool:
        """Create backup of files before moving them."""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            
            for file_path in files:
                filename = os.path.basename(file_path)
                backup_path = os.path.join(self.backup_dir, filename)
                shutil.copy2(file_path, backup_path)
            
            logger.info(f"Created backup of {len(files)} files in {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def organize_files(self, dry_run: bool = True) -> Dict[str, int]:
        """Organize scattered files into structured directories."""
        scattered_files = self.scan_scattered_files()
        
        # Count total files
        total_files = sum(len(files) for files in scattered_files.values())
        if total_files == 0:
            logger.info("No scattered files found")
            return {"organized": 0, "skipped": 0, "errors": 0}
        
        logger.info(f"Found {total_files} scattered files to organize")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be moved")
        
        stats = {"organized": 0, "skipped": 0, "errors": 0}
        
        # Group files by run ID
        all_files = []
        for file_list in scattered_files.values():
            all_files.extend(file_list)
        
        grouped_files = self.group_files_by_run_id(all_files)
        
        for run_id, files in grouped_files.items():
            if run_id == "unknown":
                logger.warning(f"Skipping {len(files)} files with unknown run ID")
                stats["skipped"] += len(files)
                continue
            
            try:
                # Create experiment directory structure
                experiment_name = f"organized_{run_id}_{datetime.now().strftime('%Y%m%d')}"
                experiment_dir = os.path.join(self.reports_dir, experiment_name)
                
                if not dry_run:
                    os.makedirs(experiment_dir, exist_ok=True)
                    os.makedirs(os.path.join(experiment_dir, "figs"), exist_ok=True)
                    os.makedirs(os.path.join(experiment_dir, "data"), exist_ok=True)
                    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
                
                # Move files to appropriate subdirectories
                for file_path in files:
                    filename = os.path.basename(file_path)
                    
                    if filename.endswith('.png'):
                        target_dir = os.path.join(experiment_dir, "figs")
                    elif filename.endswith('.csv'):
                        target_dir = os.path.join(experiment_dir, "data")
                    elif filename.endswith(('.txt', '.json')):
                        target_dir = os.path.join(experiment_dir, "logs")
                    else:
                        target_dir = os.path.join(experiment_dir, "logs")
                    
                    target_path = os.path.join(target_dir, filename)
                    
                    if not dry_run:
                        shutil.move(file_path, target_path)
                        logger.info(f"Moved {file_path} -> {target_path}")
                    else:
                        logger.info(f"Would move {file_path} -> {target_path}")
                    
                    stats["organized"] += 1
                
                # Create a simple index file for the organized experiment
                if not dry_run:
                    self.create_simple_index(experiment_dir, run_id, files)
                
            except Exception as e:
                logger.error(f"Error organizing files for run {run_id}: {e}")
                stats["errors"] += len(files)
        
        return stats
    
    def create_simple_index(self, experiment_dir: str, run_id: str, files: List[str]):
        """Create a simple index file for organized experiment."""
        index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Organized Experiment - {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .file-list {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .file-item {{ margin: 10px 0; }}
        .file-item a {{ text-decoration: none; color: #007acc; }}
    </style>
</head>
<body>
    <h1>Organized Experiment: {run_id}</h1>
    <p>This experiment was automatically organized from scattered files.</p>
    
    <h2>Files:</h2>
    <div class="file-list">
"""
        
        for file_path in files:
            filename = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, experiment_dir)
            index_content += f'        <div class="file-item"><a href="{relative_path}">{filename}</a></div>\n'
        
        index_content += """
    </div>
    
    <p><em>Generated by ReportCleaner on {}</em></p>
</body>
</html>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        index_path = os.path.join(experiment_dir, f"index_{run_id}.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        logger.info(f"Created index file: {index_path}")
    
    def cleanup_empty_directories(self, dry_run: bool = True):
        """Remove empty directories after organizing files."""
        if not os.path.exists(self.reports_dir):
            return
        
        removed_count = 0
        
        for root, dirs, files in os.walk(self.reports_dir, topdown=False):
            # Skip backup directory
            if 'backup' in root:
                continue
                
            # Skip if directory contains subdirectories
            if dirs:
                continue
            
            # Skip if directory contains files
            if files:
                continue
            
            # Skip if it's the root reports directory
            if root == self.reports_dir:
                continue
            
            if not dry_run:
                os.rmdir(root)
                logger.info(f"Removed empty directory: {root}")
            else:
                logger.info(f"Would remove empty directory: {root}")
            
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} empty directories")
        else:
            logger.info("No empty directories found")

@click.group()
def main():
    """Report cleanup and organization tool."""
    pass

@main.command()
@click.option('--reports-dir', default='reports', help='Reports directory to clean up')
@click.option('--dry-run', is_flag=True, help='Show what would be done without actually doing it')
def scan(reports_dir, dry_run):
    """Scan for scattered report files."""
    cleaner = ReportCleaner(reports_dir)
    scattered_files = cleaner.scan_scattered_files()
    
    total_files = sum(len(files) for files in scattered_files.values())
    
    click.echo(f"Found {total_files} scattered files:")
    click.echo(f"  PNG files: {len(scattered_files['png_files'])}")
    click.echo(f"  CSV files: {len(scattered_files['csv_files'])}")
    click.echo(f"  TXT files: {len(scattered_files['txt_files'])}")
    click.echo(f"  JSON files: {len(scattered_files['json_files'])}")
    
    if total_files > 0:
        click.echo("\nSample files:")
        for file_type, files in scattered_files.items():
            if files:
                click.echo(f"  {file_type}: {files[0]}")

@main.command()
@click.option('--reports-dir', default='reports', help='Reports directory to clean up')
@click.option('--dry-run', is_flag=True, help='Show what would be done without actually doing it')
@click.option('--backup', is_flag=True, help='Create backup before organizing')
def organize(reports_dir, dry_run, backup):
    """Organize scattered files into structured directories."""
    cleaner = ReportCleaner(reports_dir)
    
    # Scan files first
    scattered_files = cleaner.scan_scattered_files()
    all_files = []
    for file_list in scattered_files.values():
        all_files.extend(file_list)
    
    if not all_files:
        click.echo("No scattered files found to organize")
        return
    
    # Create backup if requested
    if backup and not dry_run:
        click.echo("Creating backup...")
        if not cleaner.create_backup(all_files):
            click.echo("Failed to create backup. Aborting.", err=True)
            return
    
    # Organize files
    click.echo(f"Organizing {len(all_files)} files...")
    stats = cleaner.organize_files(dry_run)
    
    click.echo(f"\nOrganization complete:")
    click.echo(f"  Organized: {stats['organized']}")
    click.echo(f"  Skipped: {stats['skipped']}")
    click.echo(f"  Errors: {stats['errors']}")
    
    if not dry_run:
        # Clean up empty directories
        click.echo("\nCleaning up empty directories...")
        cleaner.cleanup_empty_directories(dry_run=False)
    
    if dry_run:
        click.echo("\nThis was a dry run. Use --no-dry-run to actually organize files.")

@main.command()
@click.option('--reports-dir', default='reports', help='Reports directory to clean up')
def cleanup(reports_dir):
    """Full cleanup: organize files and remove empty directories."""
    cleaner = ReportCleaner(reports_dir)
    
    # Scan first
    scattered_files = cleaner.scan_scattered_files()
    all_files = []
    for file_list in scattered_files.values():
        all_files.extend(file_list)
    
    if not all_files:
        click.echo("No scattered files found")
        return
    
    # Confirm action
    click.echo(f"Found {len(all_files)} scattered files to organize.")
    if not click.confirm("Do you want to proceed with organization?"):
        click.echo("Aborted.")
        return
    
    # Create backup
    click.echo("Creating backup...")
    if not cleaner.create_backup(all_files):
        click.echo("Failed to create backup. Aborting.", err=True)
        return
    
    # Organize files
    click.echo("Organizing files...")
    stats = cleaner.organize_files(dry_run=False)
    
    click.echo(f"\nOrganization complete:")
    click.echo(f"  Organized: {stats['organized']}")
    click.echo(f"  Skipped: {stats['skipped']}")
    click.echo(f"  Errors: {stats['errors']}")
    
    # Clean up empty directories
    click.echo("Cleaning up empty directories...")
    cleaner.cleanup_empty_directories(dry_run=False)
    
    click.echo(f"\nCleanup complete! Backup created in: {cleaner.backup_dir}")

if __name__ == '__main__':
    main()
