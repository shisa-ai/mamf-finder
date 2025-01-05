#!/usr/bin/env python3

import glob
import re
from pathlib import Path
import argparse
from typing import Dict, NamedTuple, List
from collections import defaultdict

class GPUSpecs(NamedTuple):
    name: str
    theoretical_tflops_bf16: float
    theoretical_tflops_fp8: float

# Mapping of GPU names to their theoretical peak TFLOPS for different dtypes
GPU_SPECS = {
    'NVIDIA H100 80GB HBM3': GPUSpecs('H100 SXM', 989, 1979),
    'NVIDIA H100-SXM': GPUSpecs('H100 SXM', 989, 1979),
    'NVIDIA H100-PCIe': GPUSpecs('H100 PCIe', 989, 1979),
    'NVIDIA A100-SXM4-80GB': GPUSpecs('A100 SXM', 312, 0),
    'NVIDIA A100-PCIE-80GB': GPUSpecs('A100 PCIe', 312, 0),
    'NVIDIA GH200': GPUSpecs('GH200 SXM', 989, 1979),
    'AMD MI300X': GPUSpecs('MI300X', 1300, 2600),
    'Intel Gaudi2': GPUSpecs('Gaudi 2', 432, 865),
    'Intel Gaudi3': GPUSpecs('Gaudi 3', 1835, 1835),
}

def parse_gpu_number(filename: str) -> int:
    """Extract GPU number from filename."""
    match = re.search(r'gpu(\d+)', Path(filename).name)
    return int(match.group(1)) if match else -1

def parse_file(filepath: str) -> Dict:
    """Parse a single result file and return relevant data."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract GPU name
        gpu_match = re.search(r"name='([^']+)'", content)
        gpu_name = gpu_match.group(1) if gpu_match else "Unknown GPU"
        
        # Extract torch version
        torch_match = re.search(r'torch=([\d.]+\+cu\d+)', content)
        torch_version = torch_match.group(1) if torch_match else "Unknown"
        
        # Extract dtype
        dtype_match = re.search(r'Dtype: torch\.([\w_]+)', content)
        dtype = dtype_match.group(1) if dtype_match else "Unknown"
        
        # Extract MAMF (max TFLOPS)
        tflops_match = re.search(r'max:\s*([\d.]+)\s*TFLOPS\s*@\s*(\d+)x(\d+)x(\d+)', content)
        if tflops_match:
            mamf = float(tflops_match.group(1))
            shape = f"{tflops_match.group(2)}x{tflops_match.group(3)}x{tflops_match.group(4)}"
        else:
            mamf = 0.0
            shape = "N/A"
        
        return {
            'gpu_name': gpu_name,
            'mamf': mamf,
            'shape': shape,
            'torch_version': torch_version,
            'dtype': dtype,
            'gpu_num': parse_gpu_number(filepath)
        }
    except Exception as e:
        print(f"Error parsing file {filepath}: {str(e)}")
        return None

def print_table_header():
    """Print the header for a results table."""
    print("| GPU# | Accelerator | MAMF | Theory | Efficiency | Best Shape MxNxK | torch ver | Notes |")
    print("|------|-------------|------|---------|------------|------------------|-----------|--------|")

def process_results(results: List[Dict], dtype: str, full_notes: str):
    """Process and print results for a specific dtype."""
    print(f"\n## Results for {dtype}")
    print_table_header()
    
    # Sort results by GPU number
    results.sort(key=lambda x: x['gpu_num'])
    
    for result in results:
        # Look up GPU specs
        gpu_specs = GPU_SPECS.get(result['gpu_name'], GPUSpecs(result['gpu_name'], 0, 0))
        
        # Get theoretical peak based on dtype
        theory = 0
        if dtype == 'bfloat16':
            theory = gpu_specs.theoretical_tflops_bf16
        elif dtype == 'float8_e4m3fn':
            theory = gpu_specs.theoretical_tflops_fp8
            
        # Calculate efficiency
        efficiency = (result['mamf'] / theory * 100) if theory else 0
        
        # Format and print table row
        print(f"| {result['gpu_num']} | {gpu_specs.name} | {result['mamf']:.1f} | {theory} | "
              f"{efficiency:.1f}% | {result['shape']} | {result['torch_version']} | {full_notes} |")

def main():
    parser = argparse.ArgumentParser(description='Parse GPU benchmark results and create markdown table')
    parser.add_argument('files', nargs='+', help='Files or glob pattern to process')
    parser.add_argument('--notes', default='ml.p5.48xlarge, benchmark v2',
                        help='Additional notes to add to table')
    args = parser.parse_args()
    
    # Handle both glob patterns and direct file lists
    if len(args.files) == 1:
        files = glob.glob(args.files[0])
    else:
        files = args.files
        
    if not files:
        print("No files found to process!")
        return
        
    # Extract machine name from the first file's path
    first_file = Path(files[0])
    machine_name = first_file.parent.name
    full_notes = f"{machine_name}, {args.notes}"
    
    # Group results by dtype
    results_by_dtype = defaultdict(list)
    for filepath in files:
        data = parse_file(filepath)
        if data:
            results_by_dtype[data['dtype']].append(data)
    
    # Process and print results for each dtype
    for dtype in sorted(results_by_dtype.keys()):
        process_results(results_by_dtype[dtype], dtype, full_notes)

if __name__ == '__main__':
    main()
