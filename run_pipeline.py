#!/usr/bin/env python
"""
Diffusion Policy Pipeline - Complete Automation Script
======================================================
Executes all 6 steps sequentially for MuJoCo manipulation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional


class Colors:
    """ANSI color codes"""
    HEADER = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_step(step_num: int, text: str):
    """Print step information"""
    print(f"{Colors.WARNING}[Step {step_num}] {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[SUCCESS] {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")


def run_command(cmd: list, description: str = "") -> bool:
    """Run a command and return success status"""
    try:
        if description:
            print(description)
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False


def check_file_exists(filepath: str, description: str = "") -> bool:
    """Check if a file exists"""
    if not Path(filepath).exists():
        print_error(f"File not found: {filepath}" + (f" ({description})" if description else ""))
        return False
    return True


def check_python():
    """Check Python installation"""
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Python version: {result.stdout.strip()}")
        return True
    except Exception as e:
        print_error(f"Python check failed: {e}")
        return False


def install_dependencies() -> bool:
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    packages = ["mujoco", "gymnasium", "numpy", "torch", "tqdm", "matplotlib"]
    print(f"Installing: {', '.join(packages)}")
    
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    return run_command(cmd)


# ============================================================
# Step Functions
# ============================================================

def step1_environment_testing() -> bool:
    """Step 1: Environment Standalone Testing"""
    print_header("Step 1: Environment Standalone Testing")
    
    if not check_file_exists("manipulator_env.py", "environment wrapper"):
        return False
    
    print_step(1, "Testing MuJoCo environment with Gymnasium")
    
    test_code = """
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
"""
    
    success = run_command([sys.executable, "-c", test_code])
    if success:
        print_success("Step 1 completed")
    return success


def step2_data_collection() -> bool:
    """Step 2: Data Collection"""
    print_header("Step 2: Data Collection")
    
    if not check_file_exists("collect_trajectories.py", "data collection script"):
        return False
    
    print_step(2, "Collecting trajectories from 500 episodes")
    success = run_command([sys.executable, "collect_trajectories.py"])
    if success:
        print_success("Step 2 completed")
    return success


def step3_policy_training() -> bool:
    """Step 3: Diffusion Policy Training"""
    print_header("Step 3: Diffusion Policy Training")
    
    if not check_file_exists("train_diffusion_policy.py", "training script"):
        return False
    
    if not check_file_exists("data/trajectories.npz", "training data"):
        print_error("Run Step 2 first to collect data")
        return False
    
    print_step(3, "Training TemporalUNet + DDPM model (15 epochs)")
    success = run_command([sys.executable, "train_diffusion_policy.py"])
    if success:
        print_success("Step 3 completed")
    return success


def step4_policy_validation() -> bool:
    """Step 4: Policy Validation in Simulation"""
    print_header("Step 4: Policy Validation in Simulation")
    
    if not check_file_exists("validate_policy.py", "validation script"):
        return False
    
    if not check_file_exists("models/diffusion_policy.pt", "trained model"):
        print_error("Run Step 3 first to train the model")
        return False
    
    print_step(4, "Validating policy (10 basic + 5 DR episodes)")
    success = run_command([sys.executable, "validate_policy.py"])
    if success:
        print_success("Step 4 completed")
    return success


def step5_policy_node() -> bool:
    """Step 5: Policy Node Integration"""
    print_header("Step 5: Policy Node Integration")
    
    if not check_file_exists("policy_node.py", "policy node script"):
        return False
    
    if not check_file_exists("models/diffusion_policy.pt", "trained model"):
        print_error("Run Step 3 first to train the model")
        return False
    
    print_step(5, "Running policy in simulation (3 episodes)")
    success = run_command([sys.executable, "policy_node.py"])
    if success:
        print_success("Step 5 completed")
    return success


def step6_domain_randomization() -> bool:
    """Step 6: Domain Randomization Retraining"""
    print_header("Step 6: Domain Randomization Retraining")
    
    if not check_file_exists("train_dr_policy.py", "DR training script"):
        return False
    
    if not check_file_exists("models/normalization.json", "normalization parameters"):
        print_error("Run Step 3 first to create normalization parameters")
        return False
    
    print_step(6, "Collecting DR data and training (30 epochs)")
    success = run_command([sys.executable, "train_dr_policy.py"])
    if success:
        print_success("Step 6 completed")
    return success


# ============================================================
# Main Menu
# ============================================================

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*60)
    print("Diffusion Policy Pipeline - Execute Steps")
    print("="*60)
    print("1. Step 1: Environment Testing")
    print("2. Step 2: Data Collection")
    print("3. Step 3: Policy Training")
    print("4. Step 4: Policy Validation")
    print("5. Step 5: Policy Node Integration")
    print("6. Step 6: Domain Randomization")
    print("7. Run All Steps (1-6)")
    print("8. Install Dependencies Only")
    print("0. Exit")
    print("="*60)


def run_all_steps() -> bool:
    """Run all steps in sequence"""
    steps = [
        step1_environment_testing,
        step2_data_collection,
        step3_policy_training,
        step4_policy_validation,
        step5_policy_node,
        step6_domain_randomization,
    ]
    
    for step_func in steps:
        if not step_func():
            print_error(f"Pipeline stopped at {step_func.__name__}")
            return False
    
    return True


def interactive_mode():
    """Run in interactive mode"""
    while True:
        show_menu()
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            step1_environment_testing()
        elif choice == "2":
            step2_data_collection()
        elif choice == "3":
            step3_policy_training()
        elif choice == "4":
            step4_policy_validation()
        elif choice == "5":
            step5_policy_node()
        elif choice == "6":
            step6_domain_randomization()
        elif choice == "7":
            install_dependencies()
            if run_all_steps():
                print_header("ALL STEPS COMPLETED SUCCESSFULLY!")
        elif choice == "8":
            install_dependencies()
        elif choice == "0":
            print("Exiting...")
            sys.exit(0)
        else:
            print_error("Invalid option. Please try again.")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diffusion Policy Pipeline - Automation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py all       # Run all steps
  python run_pipeline.py step1     # Run only step 1
  python run_pipeline.py deps      # Install dependencies only
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["all", "deps", "step1", "step2", "step3", "step4", "step5", "step6"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    print_header("Diffusion Policy Pipeline - Automation Script")
    
    if not check_python():
        sys.exit(1)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Execute based on command
    if args.command is None:
        # Interactive mode
        interactive_mode()
    elif args.command == "all":
        install_dependencies()
        if run_all_steps():
            print_header("ALL STEPS COMPLETED SUCCESSFULLY!")
        else:
            sys.exit(1)
    elif args.command == "deps":
        install_dependencies()
    elif args.command == "step1":
        step1_environment_testing()
    elif args.command == "step2":
        step2_data_collection()
    elif args.command == "step3":
        step3_policy_training()
    elif args.command == "step4":
        step4_policy_validation()
    elif args.command == "step5":
        step5_policy_node()
    elif args.command == "step6":
        step6_domain_randomization()


if __name__ == "__main__":
    main()
