#!/usr/bin/env bash
# Run: ./scripts/uv_run_with_cuda.sh python train_fm.py configs/config_tinystories_unconditional.yaml
# Use this when you use uv (no conda) and Triton needs CUDA_HOME to compile.
# One-time: install CUDA toolkit for uv (see README "UV without conda").

set -e
cd "$(dirname "$0")/.."

# 1) Use CUDA_HOME if already set
if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/include/cuda.h" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
  exec uv run "$@"
fi

# 2) Try to find CUDA from uv env (pip-installed nvidia-cuda-nvcc / cuda-toolkit)
UV_CUDA=$(uv run python -c "
import os, sys
for p in sys.path:
    if not os.path.isdir(p):
        continue
    for root, _, files in os.walk(p):
        if 'cuda.h' not in files:
            continue
        # cuda.h at root; CUDA_HOME = parent of 'include' or parent of root
        if os.path.basename(root) == 'include':
            print(os.path.dirname(root))
        else:
            print(root)
        sys.exit(0)
" 2>/dev/null || true)

if [ -n "$UV_CUDA" ]; then
  export CUDA_HOME="$UV_CUDA"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
  export CPATH="$CUDA_HOME/include:$CUDA_HOME:$CPATH"
  export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH"
  exec uv run "$@"
fi

# 3) Not found: run anyway (Triton may fail with clear error)
echo "Warning: CUDA_HOME not set and no pip cuda-toolkit found. Triton compile may fail." >&2
echo "See README 'UV without conda' to install cuda-toolkit or set CUDA_HOME." >&2
exec uv run "$@"
