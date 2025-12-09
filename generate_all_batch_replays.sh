#!/bin/bash

# Generate replays and GIFs for all batch results in results/mcts-data
# Author: Andy Dao

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="results/mcts-data"
GENERATE_GIFS=${GENERATE_GIFS:-true}  # Set to "false" to skip GIF generation

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

print_processing() {
    echo -e "${CYAN}▶ $1${NC}"
}

# Check if Python script exists
if [ ! -f "generate_batch_replays.py" ]; then
    print_error "generate_batch_replays.py not found!"
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    print_error "Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Find all JSON files in the results directory
BATCH_FILES=$(find "$RESULTS_DIR" -name "*.json" -type f | sort)

if [ -z "$BATCH_FILES" ]; then
    print_error "No JSON files found in $RESULTS_DIR"
    exit 1
fi

# Count files
FILE_COUNT=$(echo "$BATCH_FILES" | wc -l)
print_header "Batch Replay Generator"
echo "Results directory: $RESULTS_DIR"
echo "Found $FILE_COUNT batch result file(s)"
echo "Generate GIFs: $GENERATE_GIFS"
echo ""

# Process each batch file
current_file=0
for batch_file in $BATCH_FILES; do
    current_file=$((current_file + 1))
    filename=$(basename "$batch_file")
    
    print_processing "Processing file $current_file/$FILE_COUNT: $filename"
    
    # Run the Python script
    if [ "$GENERATE_GIFS" = "true" ]; then
        if python3 generate_batch_replays.py -f "$batch_file"; then
            print_success "Completed: $filename"
        else
            print_error "Failed: $filename"
        fi
    else
        if python3 generate_batch_replays.py -f "$batch_file" --no-gifs; then
            print_success "Completed: $filename"
        else
            print_error "Failed: $filename"
        fi
    fi
    
    echo ""
done

print_header "Batch Replay Generation Complete"
print_info "Replays saved to: replays/{batch_file_name}/"
if [ "$GENERATE_GIFS" = "true" ]; then
    print_info "GIFs saved alongside replay JSON files"
fi

