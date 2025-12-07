#!/bin/bash

# Heuristic Fine-Tuning Test Suite
# Runs batch simulations for MCTS vs Random comparisons and complexity analysis
# Author: Sierra Andrews, Andy Dao

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_GAMES=${NUM_GAMES:-100}
OUTPUT_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_run_${TIMESTAMP}.log"

# Test configurations
declare -a TEST_CONFIGS=(
    "mcts-random.json:MCTS vs Random (MCTS first)"
    "random-mcts.json:Random vs MCTS (Random first)"
    "mcts-mcts.json:MCTS vs MCTS (Complexity analysis)"
    "random-random.json:Random vs Random (Baseline)"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if Python script exists
if [ ! -f "batch_run.py" ]; then
    print_error "batch_run.py not found!"
    exit 1
fi

# Check if JSON configs exist
for config_info in "${TEST_CONFIGS[@]}"; do
    config_file="${config_info%%:*}"
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
done

print_header "Heuristic Fine-Tuning Test Suite"
echo "Timestamp: $TIMESTAMP"
echo "Number of games per test: $NUM_GAMES"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Start logging
log "Starting heuristic fine-tuning test suite"
log "Configuration: NUM_GAMES=$NUM_GAMES, OUTPUT_DIR=$OUTPUT_DIR"

# Run each test configuration
total_tests=${#TEST_CONFIGS[@]}
current_test=0

for config_info in "${TEST_CONFIGS[@]}"; do
    current_test=$((current_test + 1))
    config_file="${config_info%%:*}"
    test_name="${config_info##*:}"
    
    print_header "Test $current_test/$total_tests: $test_name"
    log "Running test: $test_name (config: $config_file)"
    
    start_time=$(date +%s)
    
    # Run the batch simulation
    if python3 batch_run.py -f "$config_file" -n "$NUM_GAMES" -o "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "Completed: $test_name (Duration: ${duration}s)"
        log "Test completed successfully in ${duration} seconds"
    else
        print_error "Failed: $test_name"
        log "Test failed: $test_name"
        # Continue with other tests even if one fails
    fi
    
    echo ""
done

# Summary
print_header "Test Suite Summary"
log "All tests completed"

# List all result files created
echo "Result files generated:"
find "$OUTPUT_DIR" -name "batch_results_*.json" -type f -newer "$LOG_FILE" 2>/dev/null | sort | while read -r file; do
    filename=$(basename "$file")
    filesize=$(du -h "$file" | cut -f1)
    print_info "  $filename ($filesize)"
done

echo ""
print_success "All tests completed! Check $OUTPUT_DIR for detailed results."
print_info "Full log available at: $LOG_FILE"

# Optional: Generate a summary report
if command -v python3 &> /dev/null; then
    echo ""
    print_info "Generating summary report..."
    python3 << EOF
import json
import glob
import os
from datetime import datetime

output_dir = "$OUTPUT_DIR"
log_file = "$LOG_FILE"

# Find all result files from this run
result_files = sorted(glob.glob(os.path.join(output_dir, "batch_results_*.json")))

if result_files:
    # Filter to files created after log file
    log_mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else 0
    recent_files = [f for f in result_files if os.path.getmtime(f) >= log_mtime]
    
    if recent_files:
        print("\n=== Quick Summary ===")
        for result_file in recent_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    config = data.get('config', {})
                    agents = config.get('agents', [])
                    black_agent = agents[0][0] if agents else 'unknown'
                    white_agent = agents[1][0] if len(agents) > 1 else 'unknown'
                    summary = data.get('summary', {})
                    
                    print(f"\n{black_agent} (Black) vs {white_agent} (White):")
                    print(f"  Black wins: {data.get('black_wins', 0)} ({summary.get('black_win_rate', 0):.1f}%)")
                    print(f"  White wins: {data.get('white_wins', 0)} ({summary.get('white_win_rate', 0):.1f}%)")
                    print(f"  Draws: {data.get('draws', 0)} ({summary.get('draw_rate', 0):.1f}%)")
                    if 'avg_move_time_black' in summary:
                        print(f"  Avg move time - Black: {summary['avg_move_time_black']:.4f}s")
                    if 'avg_move_time_white' in summary:
                        print(f"  Avg move time - White: {summary['avg_move_time_white']:.4f}s")
                    print(f"  Peak memory: {summary.get('peak_memory_usage_mb', 0):.2f} MB")
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
EOF
fi

echo ""
print_header "Test Suite Complete"

