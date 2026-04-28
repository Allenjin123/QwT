#!/usr/bin/env bash
# Build & install mmcv from source against the *current* conda env's torch
# stack, so its CUDA ops (notably soft_nms) are ABI-compatible.
#
# Why this exists:
#   - Prebuilt mmcv wheels at openmmlab.com only target torch ≤2.3 / cu ≤12.1.
#     Installing them against torch ≥2.5 produces an `_ext.so` whose libtorch
#     symbols (e.g. `c10::Error`) don't resolve → ImportError at runtime.
#   - The eva-det README's warning that mmcv "cannot be built on Blackwell"
#     was specific to mmcv-full 1.6.1 + the locked old toolchain. mmcv 2.x
#     compiles fine on cu12.6+ / cu12.8 across A100, H100, Blackwell when
#     TORCH_CUDA_ARCH_LIST is set right and ninja is on PATH.
#
# Usage:
#   ./install_mmcv.sh                           # auto-detect arch via torch.cuda
#   TORCH_CUDA_ARCH_LIST="8.0;9.0" ./install_mmcv.sh
#   CONDA_ENV=qwt_d2 MAX_JOBS=24 ./install_mmcv.sh
#
# Recognised arch codes:
#   7.5  Turing  (T4, RTX 20xx)
#   8.0  Ampere  (A100)
#   8.6  Ampere  (A6000, RTX 30xx)
#   8.9  Ada     (L40, RTX 40xx)
#   9.0  Hopper  (H100)
#   10.0 Blackwell datacentre  (B100, B200)
#   12.0 Blackwell consumer    (RTX 50xx)
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-qwt_d2}"
MAX_JOBS="${MAX_JOBS:-16}"
MMCV_VERSION="${MMCV_VERSION:-mmcv}"   # pinnable: e.g. MMCV_VERSION="mmcv==2.2.0"

# 1. Activate conda env so its bin/ is on PATH (ninja, python, pip).
if ! command -v conda >/dev/null; then
  echo "[install_mmcv] conda not found in PATH" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PY="$(which python)"
PIP="$(which pip)"
if [[ "$PY" != *"$CONDA_ENV"* ]]; then
  echo "[install_mmcv] expected python under env '$CONDA_ENV', got: $PY" >&2; exit 1
fi
echo "[install_mmcv] env=$CONDA_ENV  python=$PY"

# 2. Module-load CUDA + GCC if available, but PUT conda's bin BACK in front so
#    ninja stays visible to torch.utils.cpp_extension.
module load cuda/12.8.1 gcc/11.2.0 2>/dev/null || true
export PATH="$(conda info --base)/envs/${CONDA_ENV}/bin:${PATH}"
echo "[install_mmcv] gcc=$(gcc --version | head -1)"
echo "[install_mmcv] nvcc=$(nvcc --version | tail -1)"
echo "[install_mmcv] ninja=$(ninja --version 2>/dev/null || echo MISSING)"
"$PY" -c "from torch.utils.cpp_extension import is_ninja_available; print(f'[install_mmcv] torch sees ninja: {is_ninja_available()}')"
"$PY" -c "import torch; print(f'[install_mmcv] torch={torch.__version__}  cuda={torch.version.cuda}')"

# 3. Ensure build deps (mmcv source build needs setuptools, wheel, ninja).
"$PIP" install -q setuptools wheel ninja

# 4. Decide TORCH_CUDA_ARCH_LIST. User override wins; otherwise auto-detect
#    from the GPUs visible to torch and map to the canonical codes above.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$("$PY" -c "
import torch
arches = set()
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    arches.add(f'{cap[0]}.{cap[1]}')
print(';'.join(sorted(arches)))
")"
fi
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
  echo "[install_mmcv] no GPUs visible and no TORCH_CUDA_ARCH_LIST set" >&2; exit 1
fi
export TORCH_CUDA_ARCH_LIST
echo "[install_mmcv] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"

# 5. Drop any prior install (likely the broken cu12.1 prebuilt wheel).
"$PIP" uninstall -y mmcv mmcv-full mmcv-lite 2>/dev/null || true

# 6. Source build. --no-build-isolation forces use of *this* env's torch +
#    setuptools instead of pip spinning up an empty build venv that wouldn't
#    even have torch available.
echo "[install_mmcv] starting build (this should take 3-8 min with ninja j${MAX_JOBS})..."
t0=$SECONDS
MAX_JOBS="$MAX_JOBS" MMCV_WITH_OPS=1 \
  "$PIP" install --no-build-isolation -v "$MMCV_VERSION" 2>&1 \
  | tee /tmp/mmcv_install.log \
  | grep -E --line-buffered \
      "Collecting mmcv|Successfully installed|building 'mmcv|Building wheel|Created wheel|^FAILED|^ERROR|fatal error|error:"
echo "[install_mmcv] build done in $((SECONDS - t0))s"

# 7. Smoke test: import + a tiny CUDA soft_nms call.
"$PY" - <<'PY'
import torch
from mmcv.ops import soft_nms
boxes = torch.tensor([[0,0,10,10],[1,1,11,11],[50,50,60,60.]], device='cuda')
scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
dets, keep = soft_nms(boxes=boxes, scores=scores, iou_threshold=0.3,
                      sigma=0.5, min_score=1e-3, method='linear')
print(f"[install_mmcv] soft_nms OK: keep={keep.tolist()} "
      f"scores={[round(float(x), 4) for x in dets[:, 4].tolist()]}")
PY
echo "[install_mmcv] DONE"
