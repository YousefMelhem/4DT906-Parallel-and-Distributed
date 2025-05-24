#!/bin/bash

# Compile all versions
echo "Compiling all versions..."
nvcc -O3 matmul_naive.cu -o compiled/matmul_naive
nvcc -O3 matmul_tiled_basic.cu -o compiled/matmul_tiled_basic
nvcc -O3 matmulCache_Tiled.cu -o compiled/matmul_tiled_optimized

# Run benchmarks
echo -e "\nRunning benchmarks...\n"

echo "=== Naive Implementation ==="
./compiled/matmul_naive



echo -e "\n=== Basic Tiled Implementation ==="
./compiled/matmul_tiled_basic

echo -e "\n=== Optimized Tiled Implementation ==="
./compiled/matmul_tiled_optimized

echo -e "\n=== cupy_cuBLAS ==="
python3 cu.py

echo -e "\n=== Performance Summary ==="
echo "Implementation | Matrix Size | Time (ms) | GFLOPS"
echo "------------------------------------------------" 