@echo off
REM Run full pipeline on Windows server: install deps, collect, train diffusion, validate, train PPO, evaluate, record videos
SET ROOT_DIR=%~dp0
cd /d %ROOT_DIR%

REM Optional: set environment variables before running
REM   set CONDA_ENV=myenv
REM   set VENV_PATH=C:\path\to\venv

IF DEFINED CONDA_ENV (
  echo Activating conda environment: %CONDA_ENV%
  call conda activate %CONDA_ENV% || echo "conda activation failed or conda not initialized"
) ELSE IF DEFINED VENV_PATH (
  echo Activating virtualenv at: %VENV_PATH%
  if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
  ) else (
    echo Virtualenv activate script not found at %VENV_PATH%\Scripts\activate.bat. Skipping venv activation.
  )
) ELSE (
  echo No CONDA_ENV or VENV_PATH specified: using current Python environment
)

echo [1/10] Installing Python dependencies (may take a while)
python -m pip install --upgrade pip
python -m pip install mujoco gymnasium numpy torch tqdm matplotlib imageio imageio-ffmpeg scipy

echo [2/10] (Optional) Visualize manipulator (may fail in headless servers)
python visualize_manipulator.py || echo visualize failed

echo [3/10] Data collection (Step 2)
python run_pipeline.py step2

echo [4/10] Train diffusion policy (Step 3)
python run_pipeline.py step3

echo [5/10] Validate diffusion policy (Step 4)
python run_pipeline.py step4

echo [6/10] Train PPO baseline
python train_ppo.py --total_timesteps 50000 || echo ppo train failed

echo [7/10] Evaluate and compare
python evaluate_compare.py --episodes 10

echo [8/10] Record videos of 3 episodes per policy (if rendering available)
python record_episodes.py --policy diffusion --model models/diffusion_policy.pt --norm models/normalization.json --episodes 3 || echo record diffusion failed
python record_episodes.py --policy ppo --model models/ppo_policy.pt --episodes 3 || echo record ppo failed

echo [9/10] Pack results
if not exist results mkdir results

echo [10/10] Done. Results in: results\ and models\
