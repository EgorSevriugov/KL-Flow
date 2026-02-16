#!/usr/bin/env bash
# Run from repo root: bash scripts/check_cuda.sh
# Paste the output to diagnose Triton/CUDA issues.

echo "=== 1. Find cuda.h anywhere ==="
find /usr -name "cuda.h" 2>/dev/null | head -5
find "$CONDA_PREFIX" -name "cuda.h" 2>/dev/null | head -5
find /home/jovyan -name "cuda.h" 2>/dev/null | head -5

echo ""
echo "=== 2. Conda env ==="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "python: $(which python)"
python --version 2>/dev/null

echo ""
echo "=== 3. Fix: set CUDA_HOME and re-run (if cuda.h found above) ==="
echo "If cuda.h is at e.g. /path/to/cuda/include/cuda.h, run:"
echo "  export CUDA_HOME=/path/to/cuda"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "Then run training again (conda or uv)."
