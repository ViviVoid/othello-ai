#!/bin/bash

# MCTS vs Minimax Hyperparameter Test Suite
# Tests all combinations of MCTS iterations and rollouts against Minimax
# Author: Sierra Andrews, Andy Dao

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NUM_GAMES=${NUM_GAMES:-100}
OUTPUT_DIR="results/mcts-data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="mcts_test_run_${TIMESTAMP}.log"

# MCTS hyperparameter combinations
declare -a ITERATIONS=(100 1000 10000)
declare -a ROLLOUTS=(1 5 10)

# Test configurations - will be generated dynamically
declare -a TEST_CONFIGS=()

# Generate all test configurations
for iterations in "${ITERATIONS[@]}"; do
    for rollouts in "${ROLLOUTS[@]}"; do
        # MCTS first (Black)
        TEST_CONFIGS+=("game-environments/mcts-testing-environments/mcts-minimax-${iterations}-${rollouts}.json:MCTS(${iterations}i,${rollouts}r) vs Minimax (MCTS first)")
        # Minimax first (Black)
        TEST_CONFIGS+=("game-environments/mcts-testing-environments/minimax-mcts-${iterations}-${rollouts}.json:Minimax vs MCTS(${iterations}i,${rollouts}r) (Minimax first)")
    done
done

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

print_test() {
    echo -e "${CYAN}▶ $1${NC}"
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
missing_configs=0
for config_info in "${TEST_CONFIGS[@]}"; do
    config_file="${config_info%%:*}"
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        missing_configs=$((missing_configs + 1))
    fi
done

if [ $missing_configs -gt 0 ]; then
    print_error "$missing_configs configuration file(s) missing!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

print_header "MCTS vs Minimax Hyperparameter Test Suite"
echo "Timestamp: $TIMESTAMP"
echo "Number of games per test: $NUM_GAMES"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Testing ${#ITERATIONS[@]} iteration levels × ${#ROLLOUTS[@]} rollout levels × 2 player orders = ${#TEST_CONFIGS[@]} configurations"
echo ""

# Start logging
log "Starting MCTS vs Minimax hyperparameter test suite"
log "Configuration: NUM_GAMES=$NUM_GAMES, OUTPUT_DIR=$OUTPUT_DIR"
log "Iterations: ${ITERATIONS[*]}"
log "Rollouts: ${ROLLOUTS[*]}"

# Run each test configuration
total_tests=${#TEST_CONFIGS[@]}
current_test=0
total_start_time=$(date +%s)

# Group by iteration level for better organization
for iterations in "${ITERATIONS[@]}"; do
    print_header "Testing with ${iterations} iterations"
    
    for rollouts in "${ROLLOUTS[@]}"; do
        # MCTS first
        current_test=$((current_test + 1))
        config_file="game-environments/mcts-testing-environments/mcts-minimax-${iterations}-${rollouts}.json"
        test_name="MCTS(${iterations}i,${rollouts}r) vs Minimax (MCTS first)"
        
        print_test "Test $current_test/$total_tests: $test_name"
        log "Running test: $test_name (config: $config_file)"
        
        start_time=$(date +%s)
        
        if python3 batch_run.py -f "$config_file" -n "$NUM_GAMES" -o "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            print_success "Completed: $test_name (Duration: ${duration}s)"
            log "Test completed successfully in ${duration} seconds"
        else
            print_error "Failed: $test_name"
            log "Test failed: $test_name"
        fi
        
        echo ""
        
        # Minimax first
        current_test=$((current_test + 1))
        config_file="game-environments/mcts-testing-environments/minimax-mcts-${iterations}-${rollouts}.json"
        test_name="Minimax vs MCTS(${iterations}i,${rollouts}r) (Minimax first)"
        
        print_test "Test $current_test/$total_tests: $test_name"
        log "Running test: $test_name (config: $config_file)"
        
        start_time=$(date +%s)
        
        if python3 batch_run.py -f "$config_file" -n "$NUM_GAMES" -o "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            print_success "Completed: $test_name (Duration: ${duration}s)"
            log "Test completed successfully in ${duration} seconds"
        else
            print_error "Failed: $test_name"
            log "Test failed: $test_name"
        fi
        
        echo ""
    done
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Summary
print_header "Test Suite Summary"
log "All tests completed in ${total_duration} seconds"

# List all result files created
echo "Result files generated in $OUTPUT_DIR:"
find "$OUTPUT_DIR" -name "batch_results_*.json" -type f -newer "$LOG_FILE" 2>/dev/null | sort | while read -r file; do
    filename=$(basename "$file")
    filesize=$(du -h "$file" | cut -f1)
    print_info "  $filename ($filesize)"
done

echo ""
print_success "All tests completed! Check $OUTPUT_DIR for detailed results."
print_info "Full log available at: $LOG_FILE"

# Generate summary report
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
        print("\n=== MCTS Hyperparameter Test Summary ===")
        print(f"\nTotal configurations tested: {len(recent_files)}")
        print("\nResults by configuration:")
        print("-" * 80)
        
        # Group by configuration
        config_results = {}
        for result_file in recent_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    config = data.get('config', {})
                    agents = config.get('agents', [])
                    
                    # Extract MCTS parameters
                    if len(agents) >= 2:
                        if agents[0][0] == 'mcts':
                            mcts_params = f"{agents[0][1]}i-{agents[0][3]}r"
                            order = "MCTS_first"
                        elif agents[1][0] == 'mcts':
                            mcts_params = f"{agents[1][1]}i-{agents[1][3]}r"
                            order = "Minimax_first"
                        else:
                            continue
                        
                        key = f"{mcts_params}_{order}"
                        if key not in config_results:
                            config_results[key] = []
                        config_results[key].append(data)
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
        
        # Print summary for each configuration
        for key in sorted(config_results.keys()):
            results = config_results[key]
            if not results:
                continue
            
            # Aggregate statistics
            total_games = sum(r.get('num_games', 0) for r in results)
            total_black_wins = sum(r.get('black_wins', 0) for r in results)
            total_white_wins = sum(r.get('white_wins', 0) for r in results)
            total_draws = sum(r.get('draws', 0) for r in results)
            
            if total_games > 0:
                black_rate = (total_black_wins / total_games) * 100
                white_rate = (total_white_wins / total_games) * 100
                draw_rate = (total_draws / total_games) * 100
                
                # Get average move times
                avg_black_time = 0
                avg_white_time = 0
                if results[0].get('summary', {}).get('avg_move_time_black'):
                    avg_black_time = results[0]['summary']['avg_move_time_black']
                if results[0].get('summary', {}).get('avg_move_time_white'):
                    avg_white_time = results[0]['summary']['avg_move_time_white']
                
                print(f"\n{key}:")
                print(f"  Games: {total_games}")
                print(f"  Black wins: {total_black_wins} ({black_rate:.1f}%)")
                print(f"  White wins: {total_white_wins} ({white_rate:.1f}%)")
                print(f"  Draws: {total_draws} ({draw_rate:.1f}%)")
                if avg_black_time > 0:
                    print(f"  Avg move time - Black: {avg_black_time:.4f}s")
                if avg_white_time > 0:
                    print(f"  Avg move time - White: {avg_white_time:.4f}s")
        
        print("\n" + "-" * 80)
EOF
fi

echo ""
print_header "MCTS Hyperparameter Test Suite Complete"
echo "Total duration: ${total_duration} seconds ($(($total_duration / 60)) minutes)"

