#!/usr/bin/env bash
# Run full pipeline on server: install deps, collect, train diffusion, validate, train PPO, evaluate, record videos

set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

# Optional: set CONDA_ENV or VENV_PATH environment variables before running
# Example: CONDA_ENV=myenv ./run_on_server.sh
# Or: VENV_PATH=/home/user/venv ./run_on_server.sh

if [ -n "${CONDA_ENV:-}" ]; then
  echo "Activating conda environment: $CONDA_ENV"
  # Ensure conda is initialized for non-interactive shells
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "conda not found in PATH. Skipping conda activation."
  fi
elif [ -n "${VENV_PATH:-}" ]; then
  echo "Activating virtualenv at: $VENV_PATH"
  if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
  else
    echo "Virtualenv activate script not found at $VENV_PATH/bin/activate. Skipping venv activation."
  fi
else
  echo "No CONDA_ENV or VENV_PATH specified: using current Python environment"
fi

echo "[1/10] Installing Python dependencies (may take a while)"
python -m pip install --upgrade pip
python -m pip install mujoco gymnasium numpy torch tqdm matplotlib imageio imageio-ffmpeg scipy

echo "[2/10] (Optional) Visualize manipulator (headless: use MUJOCO_GL=egl or xvfb)"
# Try visualizing once; ignore errors
python visualize_manipulator.py || true

echo "[3/10] Data collection (Step 2)"
python run_pipeline.py step2

echo "[4/10] Train diffusion policy (Step 3)"
python run_pipeline.py step3

echo "[5/10] Validate diffusion policy (Step 4)"
python run_pipeline.py step4

echo "[6/10] Train PPO baseline"
python train_ppo.py --total_timesteps 50000 || true

echo "[7/10] Evaluate and compare"
python evaluate_compare.py --episodes 10

echo "[8/10] Record videos of 3 episodes per policy (if rendering available)"
python record_episodes.py --policy diffusion --model models/diffusion_policy.pt --norm models/normalization.json --episodes 3 || true
python record_episodes.py --policy ppo --model models/ppo_policy.pt --episodes 3 || true

echo "[9/10] Pack results"
mkdir -p results/videos || true

echo "[10/11] Run best checkpoint across experiment conditions"
python run_best_model_experiments.py --episodes 5 --episode-length 200

echo "[11/11] Done. Results in: results/ and models/"
