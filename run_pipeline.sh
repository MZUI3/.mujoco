#!/bin/bash

# ============================================================
# Diffusion Policy Pipeline - Complete Automation Script
# ============================================================
# Executes all 6 steps sequentially for MuJoCo manipulation
# ============================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[Step $1] $2${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Check Python installation
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    echo "Python version: $(python --version)"
}

# Install required packages
install_dependencies() {
    print_header "Installing Dependencies"
    
    echo "Installing required packages..."
    python -m pip install -q mujoco gymnasium numpy torch tqdm matplotlib
    print_success "Dependencies installed"
}

# ============================================================
# Step 1: Environment Testing
# ============================================================
run_step1() {
    print_header "Step 1: Environment Standalone Testing"
    
    if [ ! -f "manipulator_env.py" ]; then
        print_error "manipulator_env.py not found"
        return 1
    fi
    
    print_step 1 "Testing MuJoCo environment with Gymnasium"
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
    
    [ $? -eq 0 ] && print_success "Step 1 completed" || return 1
}

# ============================================================
# Step 2: Data Collection
# ============================================================
run_step2() {
    print_header "Step 2: Data Collection"
    
    if [ ! -f "collect_trajectories.py" ]; then
        print_error "collect_trajectories.py not found"
        return 1
    fi
    
    print_step 2 "Collecting trajectories from 500 episodes"
    python collect_trajectories.py
    
    [ $? -eq 0 ] && print_success "Step 2 completed" || return 1
}

# ============================================================
# Step 3: Diffusion Policy Training
# ============================================================
run_step3() {
    print_header "Step 3: Diffusion Policy Training"
    
    if [ ! -f "train_diffusion_policy.py" ]; then
        print_error "train_diffusion_policy.py not found"
        return 1
    fi
    
    if [ ! -f "data/trajectories.npz" ]; then
        print_error "data/trajectories.npz not found. Run Step 2 first"
        return 1
    fi
    
    print_step 3 "Training TemporalUNet + DDPM model (15 epochs)"
    python train_diffusion_policy.py
    
    [ $? -eq 0 ] && print_success "Step 3 completed" || return 1
}

# ============================================================
# Step 4: Policy Validation
# ============================================================
run_step4() {
    print_header "Step 4: Policy Validation in Simulation"
    
    if [ ! -f "validate_policy.py" ]; then
        print_error "validate_policy.py not found"
        return 1
    fi
    
    if [ ! -f "models/diffusion_policy.pt" ]; then
        print_error "models/diffusion_policy.pt not found. Run Step 3 first"
        return 1
    fi
    
    print_step 4 "Validating policy (10 basic + 5 DR episodes)"
    python validate_policy.py
    
    [ $? -eq 0 ] && print_success "Step 4 completed" || return 1
}

# ============================================================
# Step 5: Policy Node Integration
# ============================================================
run_step5() {
    print_header "Step 5: Policy Node Integration"
    
    if [ ! -f "policy_node.py" ]; then
        print_error "policy_node.py not found"
        return 1
    fi
    
    if [ ! -f "models/diffusion_policy.pt" ]; then
        print_error "models/diffusion_policy.pt not found. Run Step 3 first"
        return 1
    fi
    
    print_step 5 "Running policy in simulation (3 episodes)"
    python policy_node.py
    
    [ $? -eq 0 ] && print_success "Step 5 completed" || return 1
}

# ============================================================
# Step 6: Domain Randomization Retraining
# ============================================================
run_step6() {
    print_header "Step 6: Domain Randomization Retraining"
    
    if [ ! -f "train_dr_policy.py" ]; then
        print_error "train_dr_policy.py not found"
        return 1
    fi
    
    if [ ! -f "models/normalization.json" ]; then
        print_error "models/normalization.json not found. Run Step 3 first"
        return 1
    fi
    
    print_step 6 "Collecting DR data and training (30 epochs)"
    python train_dr_policy.py
    
    [ $? -eq 0 ] && print_success "Step 6 completed" || return 1
}

# ============================================================
# Main Menu
# ============================================================
show_menu() {
    echo ""
    echo "Diffusion Policy Pipeline - Execute Steps"
    echo "=========================================="
    echo "1. Step 1: Environment Testing"
    echo "2. Step 2: Data Collection"
    echo "3. Step 3: Policy Training"
    echo "4. Step 4: Policy Validation"
    echo "5. Step 5: Policy Node Integration"
    echo "6. Step 6: Domain Randomization"
    echo "7. Run All Steps (1-6)"
    echo "8. Install Dependencies Only"
    echo "0. Exit"
    echo ""
}

# ============================================================
# Main Execution
# ============================================================
main() {
    print_header "Diffusion Policy Pipeline - Automation Script"
    
    check_python
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            
            case $choice in
                1) run_step1 ;;
                2) run_step2 ;;
                3) run_step3 ;;
                4) run_step4 ;;
                5) run_step5 ;;
                6) run_step6 ;;
                7)
                    install_dependencies
                    run_step1 && \
                    run_step2 && \
                    run_step3 && \
                    run_step4 && \
                    run_step5 && \
                    run_step6 && \
                    print_header "ALL STEPS COMPLETED SUCCESSFULLY!"
                    ;;
                8) install_dependencies ;;
                0) 
                    echo "Exiting..."
                    exit 0
                    ;;
                *)
                    print_error "Invalid option. Please try again."
                    ;;
            esac
        done
    else
        # Command line mode
        case "$1" in
            all)
                install_dependencies
                run_step1 && \
                run_step2 && \
                run_step3 && \
                run_step4 && \
                run_step5 && \
                run_step6 && \
                print_header "ALL STEPS COMPLETED SUCCESSFULLY!"
                ;;
            deps)
                install_dependencies
                ;;
            step1)
                run_step1
                ;;
            step2)
                run_step2
                ;;
            step3)
                run_step3
                ;;
            step4)
                run_step4
                ;;
            step5)
                run_step5
                ;;
            step6)
                run_step6
                ;;
            *)
                echo "Usage: $0 [all|deps|step1|step2|step3|step4|step5|step6]"
                echo ""
                echo "Examples:"
                echo "  $0 all         # Run all steps"
                echo "  $0 step1       # Run only step 1"
                echo "  $0 deps        # Install dependencies only"
                exit 1
                ;;
        esac
    fi
}

# Run main
main "$@"
