#!/bin/bash
# update_data.sh - Automated data update script
# This script demonstrates the data update functionality mentioned in README

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
LOG_FILE="$PROJECT_DIR/logs/data_update.log"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Default parameters
PAIRS=("ETHUSDC" "BTCUSDC" "USDCUSDT")
INTERVALS=("1h" "1d")
DAYS_BACK=7
SOURCE="binance"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs)
            IFS=',' read -ra PAIRS <<< "$2"
            shift 2
            ;;
        --intervals)
            IFS=',' read -ra INTERVALS <<< "$2"
            shift 2
            ;;
        --days)
            DAYS_BACK="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --pairs PAIRS        Comma-separated list of trading pairs (default: ETHUSDC,BTCUSDC,USDCUSDT)"
            echo "  --intervals INTVLS   Comma-separated list of intervals (default: 1h,1d)"
            echo "  --days DAYS          Number of days back to fetch (default: 7)"
            echo "  --source SOURCE      Data source: binance or kraken (default: binance)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate date range
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "$DAYS_BACK days ago" +%Y-%m-%d)

log "Starting data update for $SOURCE"
log "Date range: $START_DATE to $END_DATE"
log "Pairs: ${PAIRS[*]}"
log "Intervals: ${INTERVALS[*]}"

# Change to project directory
cd "$PROJECT_DIR"

# Update data for each pair and interval
for pair in "${PAIRS[@]}"; do
    for interval in "${INTERVALS[@]}"; do
        output_file="data/${pair}_${interval}_$(date +%Y%m%d).csv"
        
        log "Updating $pair $interval data..."
        
        if python cli.py fetch \
            --source "$SOURCE" \
            --symbol "$pair" \
            --interval "$interval" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --out "$output_file"; then
            
            log "✅ Successfully updated $pair $interval data: $output_file"
            
            # Verify the downloaded data
            if [ -f "$output_file" ]; then
                record_count=$(wc -l < "$output_file")
                file_size=$(du -h "$output_file" | cut -f1)
                log "   Records: $((record_count - 1)) (excluding header)"
                log "   File size: $file_size"
            fi
        else
            log "❌ Failed to update $pair $interval data"
        fi
        
        # Add delay to respect rate limits
        if [ "$SOURCE" = "binance" ]; then
            sleep 1  # Binance: 1200 requests/min
        elif [ "$SOURCE" = "kraken" ]; then
            sleep 7  # Kraken: 15 requests/10s
        fi
    done
done

log "Data update completed"
log "Total files in data directory: $(find "$DATA_DIR" -name "*.csv" | wc -l)"

# Optional: Clean up old files (older than 30 days)
log "Cleaning up old data files..."
find "$DATA_DIR" -name "*.csv" -mtime +30 -exec rm -f {} \;
log "Cleanup completed"

log "Script finished successfully"
