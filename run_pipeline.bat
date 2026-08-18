@echo off
REM ============================================================
REM Diffusion Policy Pipeline - Complete Automation Script
REM ============================================================
REM Executes all 6 steps sequentially for MuJoCo manipulation
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Colors (requires Windows 10+)
for /F "tokens=1,2 delims=#" %%a in ('prompt #$H#$E# & echo prompt #$H#$E# ^| cmd') do (
    set "BS=%%a"
)

REM Color codes
set "RESET=%%b"
set "GREEN=[92m"
set "RED=[91m"
set "BLUE=[94m"
set "YELLOW=[93m"

REM Logging functions
setlocal enabledelayedexpansion

:print_header
    echo.
    echo ============================================================
    echo %~1
    echo ============================================================
    echo.
    exit /b 0

:print_step
    echo [Step %~1] %~2
    exit /b 0

:print_success
    echo [SUCCESS] %~1
    exit /b 0

:print_error
    echo [ERROR] %~1
    exit /b 0

REM ============================================================
REM Check Python installation
REM ============================================================
:check_python
    python --version >nul 2>&1
    if errorlevel 1 (
        call :print_error "Python not found. Please install Python 3.8+"
        exit /b 1
    )
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo Python version: !PYTHON_VERSION!
    exit /b 0

REM ============================================================
REM Install dependencies
REM ============================================================
:install_dependencies
    call :print_header "Installing Dependencies"
    
    echo Installing required packages...
    python -m pip install -q mujoco gymnasium numpy torch tqdm matplotlib
    if errorlevel 1 (
        call :print_error "Failed to install dependencies"
        exit /b 1
    )
    call :print_success "Dependencies installed"
    exit /b 0

REM ============================================================
REM Step 1: Environment Testing
REM ============================================================
:run_step1
    call :print_header "Step 1: Environment Standalone Testing"
    
    if not exist "manipulator_env.py" (
        call :print_error "manipulator_env.py not found"
        exit /b 1
    )
    
    call :print_step 1 "Testing MuJoCo environment with Gymnasium"
    
    python -c "
from manipulator_env import DomainRandomizedEnv
import numpy as np

env = DomainRandomizedEnv(randomize=False, render_mode=None)
print('[Test] Resetting environment...')
obs, info = env.reset()
print(f'  Observation shape: {obs.shape}')
print(f'  Observation range: [{obs.min():.3f}, {obs.max():.3f}]')

print('[Test] Running 10 steps...')
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'  Step {step+1}: reward={reward:.4f}, obs_shape={obs.shape}')
    if terminated or truncated:
        break

env.close()
print('[OK] Environment test passed!')
"
    
    if errorlevel 1 exit /b 1
    call :print_success "Step 1 completed"
    exit /b 0

REM ============================================================
REM Step 2: Data Collection
REM ============================================================
:run_step2
    call :print_header "Step 2: Data Collection"
    
    if not exist "collect_trajectories.py" (
        call :print_error "collect_trajectories.py not found"
        exit /b 1
    )
    
    call :print_step 2 "Collecting trajectories from 500 episodes"
    python collect_trajectories.py
    if errorlevel 1 exit /b 1
    call :print_success "Step 2 completed"
    exit /b 0

REM ============================================================
REM Step 3: Diffusion Policy Training
REM ============================================================
:run_step3
    call :print_header "Step 3: Diffusion Policy Training"
    
    if not exist "train_diffusion_policy.py" (
        call :print_error "train_diffusion_policy.py not found"
        exit /b 1
    )
    
    if not exist "data\trajectories.npz" (
        call :print_error "data\trajectories.npz not found. Run Step 2 first"
        exit /b 1
    )
    
    call :print_step 3 "Training TemporalUNet + DDPM model (15 epochs)"
    python train_diffusion_policy.py
    if errorlevel 1 exit /b 1
    call :print_success "Step 3 completed"
    exit /b 0

REM ============================================================
REM Step 4: Policy Validation
REM ============================================================
:run_step4
    call :print_header "Step 4: Policy Validation in Simulation"
    
    if not exist "validate_policy.py" (
        call :print_error "validate_policy.py not found"
        exit /b 1
    )
    
    if not exist "models\diffusion_policy.pt" (
        call :print_error "models\diffusion_policy.pt not found. Run Step 3 first"
        exit /b 1
    )
    
    call :print_step 4 "Validating policy (10 basic + 5 DR episodes)"
    python validate_policy.py
    if errorlevel 1 exit /b 1
    call :print_success "Step 4 completed"
    exit /b 0

REM ============================================================
REM Step 5: Policy Node Integration
REM ============================================================
:run_step5
    call :print_header "Step 5: Policy Node Integration"
    
    if not exist "policy_node.py" (
        call :print_error "policy_node.py not found"
        exit /b 1
    )
    
    if not exist "models\diffusion_policy.pt" (
        call :print_error "models\diffusion_policy.pt not found. Run Step 3 first"
        exit /b 1
    )
    
    call :print_step 5 "Running policy in simulation (3 episodes)"
    python policy_node.py
    if errorlevel 1 exit /b 1
    call :print_success "Step 5 completed"
    exit /b 0

REM ============================================================
REM Step 6: Domain Randomization Retraining
REM ============================================================
:run_step6
    call :print_header "Step 6: Domain Randomization Retraining"
    
    if not exist "train_dr_policy.py" (
        call :print_error "train_dr_policy.py not found"
        exit /b 1
    )
    
    if not exist "models\normalization.json" (
        call :print_error "models\normalization.json not found. Run Step 3 first"
        exit /b 1
    )
    
    call :print_step 6 "Collecting DR data and training (30 epochs)"
    python train_dr_policy.py
    if errorlevel 1 exit /b 1
    call :print_success "Step 6 completed"
    exit /b 0

REM ============================================================
REM Show Menu
REM ============================================================
:show_menu
    cls
    echo.
    echo Diffusion Policy Pipeline - Execute Steps
    echo ==========================================
    echo 1. Step 1: Environment Testing
    echo 2. Step 2: Data Collection
    echo 3. Step 3: Policy Training
    echo 4. Step 4: Policy Validation
    echo 5. Step 5: Policy Node Integration
    echo 6. Step 6: Domain Randomization
    echo 7. Run All Steps (1-6)
    echo 8. Install Dependencies Only
    echo 0. Exit
    echo.
    exit /b 0

REM ============================================================
REM Main Execution
REM ============================================================
:main
    call :print_header "Diffusion Policy Pipeline - Automation Script"
    
    call :check_python
    if errorlevel 1 exit /b 1
    
    REM Check for command line arguments
    if "%~1"=="" (
        REM Interactive mode
        :interactive_loop
        call :show_menu
        set /p choice="Select option: "
        
        if "!choice!"=="1" (
            call :run_step1
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="2" (
            call :run_step2
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="3" (
            call :run_step3
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="4" (
            call :run_step4
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="5" (
            call :run_step5
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="6" (
            call :run_step6
            if not errorlevel 1 pause
            goto interactive_loop
        ) else if "!choice!"=="7" (
            call :install_dependencies
            call :run_step1 && call :run_step2 && call :run_step3 && call :run_step4 && call :run_step5 && call :run_step6
            if not errorlevel 1 (
                call :print_header "ALL STEPS COMPLETED SUCCESSFULLY!"
            )
            pause
            goto interactive_loop
        ) else if "!choice!"=="8" (
            call :install_dependencies
            pause
            goto interactive_loop
        ) else if "!choice!"=="0" (
            echo Exiting...
            exit /b 0
        ) else (
            echo Invalid option. Please try again.
            pause
            goto interactive_loop
        )
    ) else (
        REM Command line mode
        if "%~1"=="all" (
            call :install_dependencies
            call :run_step1 && call :run_step2 && call :run_step3 && call :run_step4 && call :run_step5 && call :run_step6
            if not errorlevel 1 (
                call :print_header "ALL STEPS COMPLETED SUCCESSFULLY!"
            )
        ) else if "%~1"=="deps" (
            call :install_dependencies
        ) else if "%~1"=="step1" (
            call :run_step1
        ) else if "%~1"=="step2" (
            call :run_step2
        ) else if "%~1"=="step3" (
            call :run_step3
        ) else if "%~1"=="step4" (
            call :run_step4
        ) else if "%~1"=="step5" (
            call :run_step5
        ) else if "%~1"=="step6" (
            call :run_step6
        ) else (
            echo Usage: %~0 [all^|deps^|step1^|step2^|step3^|step4^|step5^|step6]
            echo.
            echo Examples:
            echo   %~0 all      - Run all steps
            echo   %~0 step1    - Run only step 1
            echo   %~0 deps     - Install dependencies only
            exit /b 1
        )
    )
    exit /b 0

call :main %*
