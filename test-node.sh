#!/bin/bash

# Handle optional machine parameter
output_dir="."
if [ $# -eq 1 ]; then
    machine=$1
    output_dir="$machine"
    mkdir -p "$output_dir"
    echo "Output will be saved in directory: $output_dir"
fi

# Record start time
start_time=$(date +%s)
echo "Starting multi-GPU test at $(date)"

# Function to run tests for a specific GPU
run_gpu_tests() {
    local gpu_id=$1
    echo "Starting tests for GPU $gpu_id"
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Run bfloat16 test
    echo "Running bfloat16 test on GPU $gpu_id"
    ./mamf-finder.py --dtype bfloat16 \
        --m_range 0 16384 1024 \
        --n_range 0 16384 1024 \
        --k_range 0 16384 1024 \
        --output_file="$output_dir/gpu${gpu_id}-bfloat16-$(date +"%Y-%m-%d-%H-%M-%S").txt"
    
    # Run float8 test
    echo "Running float8_e4m3fn test on GPU $gpu_id"
    ./mamf-finder.py --dtype float8_e4m3fn \
        --m_range 0 16384 1024 \
        --n_range 0 16384 1024 \
        --k_range 0 16384 1024 \
        --output_file="$output_dir/gpu${gpu_id}-float8_e4m3fn-$(date +"%Y-%m-%d-%H-%M-%S").txt"
        
    echo "Completed tests for GPU $gpu_id"
}

# Array to store background process IDs
pids=()

# Launch tests for each GPU in parallel
for gpu in {0..7}; do
    run_gpu_tests $gpu &
    pids+=($!)
done

# Wait for all processes to complete
echo "Waiting for all GPU tests to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

# Calculate and display elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))

echo "All GPU tests completed at $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
