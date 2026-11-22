#!/usr/bin/env bash
set -e
echo "=== 1. Running Python FP32 Control Baseline ==="
python3 baseline_python/run_baseline.py

echo "=== 2. Building C++20 Zero-Copy INT8 Engine ==="
mkdir -p cpp_engine/build && cd cpp_engine/build
cmake .. && make -j$(nproc)

echo "=== 3. Capturing Hardware Performance Counters (perf stat) ==="
LD_LIBRARY_PATH=../../third_party/onnxruntime/lib perf stat \
  -e cycles,instructions,cache-references,cache-misses,branch-misses \
  ./railguard_engine ../../data/test_track.mp4
