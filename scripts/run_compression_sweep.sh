#!/bin/bash

# TIFF Compression Analysis Sweep Script
# This script runs the TIFF compression analysis over multiple quality levels
# and all available CT file datasets.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYSIS_SCRIPT="$SCRIPT_DIR/tiff_compression_analysis.py"
DATA_BASE_DIR="$PROJECT_ROOT/data/ct_files"
OUTPUT_BASE_DIR="$SCRIPT_DIR/compression_sweep_results"

# Quality levels to test (cq_hw values)
CQ_VALUES=(100 98 96 94 92 90 85 80)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to get a clean experiment name from folder path
get_experiment_name() {
    local folder_path="$1"
    local cq_value="$2"
    local folder_name=$(basename "$folder_path")
    
    # Clean up the folder name for wandb (remove special characters, limit length)
    local clean_name=$(echo "$folder_name" | sed 's/[^a-zA-Z0-9_-]/_/g' | cut -c1-50)
    echo "${clean_name}_cq${cq_value}"
}

# Function to run analysis for a single dataset and quality
run_single_analysis() {
    local data_path="$1"
    local cq_value="$2"
    local dataset_name=$(basename "$data_path")
    local experiment_name=$(get_experiment_name "$data_path" "$cq_value")
    local output_dir="$OUTPUT_BASE_DIR/${dataset_name}/cq_${cq_value}"
    
    print_info "Running analysis: Dataset='$dataset_name', CQ=$cq_value"
    print_info "Output directory: $output_dir"
    print_info "Experiment name: $experiment_name"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the analysis
    python "$ANALYSIS_SCRIPT" \
        --data_path "$data_path" \
        --cq_hw "$cq_value" \
        --output_dir "$output_dir" \
        --experiment_name "$experiment_name" \
        --wandb_project "tiff-compression-sweep"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Completed: $dataset_name (CQ=$cq_value)"
        return 0
    else
        print_error "Failed: $dataset_name (CQ=$cq_value) - Exit code: $exit_code"
        return 1
    fi
}

# Main execution
main() {
    print_header "TIFF Compression Analysis Sweep"
    
    # Check if analysis script exists
    if [ ! -f "$ANALYSIS_SCRIPT" ]; then
        print_error "Analysis script not found: $ANALYSIS_SCRIPT"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_BASE_DIR" ]; then
        print_error "Data directory not found: $DATA_BASE_DIR"
        exit 1
    fi
    
    # Find all subdirectories in the data folder
    print_info "Scanning for datasets in: $DATA_BASE_DIR"
    
    # Get list of directories containing TIFF files
    DATA_FOLDERS=()
    while IFS= read -r -d '' folder; do
        # Check if folder contains TIFF files
        if find "$folder" -maxdepth 1 -name "*.tif" -o -name "*.tiff" | head -1 | grep -q .; then
            DATA_FOLDERS+=("$folder")
            print_info "Found dataset: $(basename "$folder")"
        fi
    done < <(find "$DATA_BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
    
    if [ ${#DATA_FOLDERS[@]} -eq 0 ]; then
        print_error "No datasets with TIFF files found in $DATA_BASE_DIR"
        exit 1
    fi
    
    print_success "Found ${#DATA_FOLDERS[@]} datasets with TIFF files"
    print_success "Quality levels to test: ${CQ_VALUES[*]}"
    
    # Calculate total number of experiments
    total_experiments=$((${#DATA_FOLDERS[@]} * ${#CQ_VALUES[@]}))
    print_info "Total experiments to run: $total_experiments"
    
    # Create main output directory
    mkdir -p "$OUTPUT_BASE_DIR"
    
    # Initialize counters
    completed=0
    failed=0
    current=0
    
    # Log file for the sweep
    log_file="$OUTPUT_BASE_DIR/sweep_log_$(date +%Y%m%d_%H%M%S).txt"
    echo "Starting compression sweep at $(date)" > "$log_file"
    
    print_header "Starting Compression Sweep"
    
    # Loop through each dataset
    for data_folder in "${DATA_FOLDERS[@]}"; do
        dataset_name=$(basename "$data_folder")
        print_header "Processing Dataset: $dataset_name"
        
        # Loop through each quality level
        for cq_value in "${CQ_VALUES[@]}"; do
            current=$((current + 1))
            
            print_info "Progress: $current/$total_experiments"
            echo "[$current/$total_experiments] Starting: $dataset_name (CQ=$cq_value)" >> "$log_file"
            
            # Run the analysis
            if run_single_analysis "$data_folder" "$cq_value"; then
                completed=$((completed + 1))
                echo "[$current/$total_experiments] ✅ Completed: $dataset_name (CQ=$cq_value)" >> "$log_file"
            else
                failed=$((failed + 1))
                echo "[$current/$total_experiments] ❌ Failed: $dataset_name (CQ=$cq_value)" >> "$log_file"
                
                # Ask user if they want to continue on failure
                print_warning "Analysis failed. Continue with remaining experiments? (y/n)"
                read -r continue_choice
                if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
                    print_info "Stopping sweep as requested by user"
                    break 2
                fi
            fi
            
            # Small delay between experiments to prevent overwhelming the system
            sleep 2
        done
    done
    
    # Final summary
    print_header "Compression Sweep Complete"
    print_success "Completed experiments: $completed"
    if [ $failed -gt 0 ]; then
        print_warning "Failed experiments: $failed"
    fi
    print_info "Total experiments: $total_experiments"
    print_info "Results saved in: $OUTPUT_BASE_DIR"
    print_info "Log file: $log_file"
    
    # Add summary to log
    echo "" >> "$log_file"
    echo "=== FINAL SUMMARY ===" >> "$log_file"
    echo "Sweep completed at $(date)" >> "$log_file"
    echo "Completed: $completed" >> "$log_file"
    echo "Failed: $failed" >> "$log_file"
    echo "Total: $total_experiments" >> "$log_file"
    
    if [ $failed -eq 0 ]; then
        print_success "🎉 All experiments completed successfully!"
        exit 0
    else
        print_warning "Some experiments failed. Check the log file for details."
        exit 1
    fi
}

# Handle Ctrl+C gracefully
trap 'print_warning "Sweep interrupted by user"; exit 130' INT

# Run main function
main "$@"
