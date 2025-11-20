#!/usr/bin/env python3
"""Complete pipeline script for cold-start recommendation system.

This script runs the entire pipeline from data generation to model evaluation
and demo launch, providing a one-stop solution for the cold-start recommendation system.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of what the command does.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Run the complete cold-start recommendation system pipeline."""
    logger.info("Starting Cold-Start Recommendation System Pipeline")
    logger.info("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Install dependencies
    logger.info("Step 1: Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        logger.error("Failed to install dependencies. Please check requirements.txt")
        sys.exit(1)
    
    # Step 2: Generate data
    logger.info("Step 2: Generating synthetic data...")
    if not run_command("python scripts/generate_data.py", "Generating synthetic data"):
        logger.error("Failed to generate data")
        sys.exit(1)
    
    # Step 3: Train and evaluate models
    logger.info("Step 3: Training and evaluating models...")
    if not run_command("python scripts/train_evaluate.py", "Training and evaluating models"):
        logger.error("Failed to train/evaluate models")
        sys.exit(1)
    
    # Step 4: Run tests
    logger.info("Step 4: Running unit tests...")
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        logger.warning("Some tests failed, but continuing...")
    
    # Step 5: Code quality checks
    logger.info("Step 5: Running code quality checks...")
    
    # Black formatting check
    if not run_command("python -m black --check src/ scripts/ tests/", "Checking code formatting"):
        logger.warning("Code formatting issues found. Run 'black src/ scripts/ tests/' to fix.")
    
    # Ruff linting
    if not run_command("python -m ruff check src/ scripts/ tests/", "Running linting"):
        logger.warning("Linting issues found. Run 'ruff check src/ scripts/ tests/' to see details.")
    
    # MyPy type checking
    if not run_command("python -m mypy src/", "Running type checking"):
        logger.warning("Type checking issues found. Run 'mypy src/' to see details.")
    
    # Step 6: Display results
    logger.info("Step 6: Displaying results...")
    
    # Check if results exist
    results_file = Path("models/model_results.csv")
    leaderboard_file = Path("models/leaderboard.csv")
    
    if results_file.exists():
        logger.info("Model evaluation results:")
        try:
            import pandas as pd
            results_df = pd.read_csv(results_file)
            print("\n" + "=" * 80)
            print("MODEL EVALUATION RESULTS")
            print("=" * 80)
            print(results_df.to_string(index=False))
        except Exception as e:
            logger.warning(f"Could not display results: {e}")
    
    if leaderboard_file.exists():
        logger.info("Model leaderboard:")
        try:
            import pandas as pd
            leaderboard_df = pd.read_csv(leaderboard_file)
            print("\n" + "=" * 80)
            print("MODEL LEADERBOARD")
            print("=" * 80)
            print(leaderboard_df.to_string(index=False))
        except Exception as e:
            logger.warning(f"Could not display leaderboard: {e}")
    
    # Step 7: Launch demo (optional)
    logger.info("Step 7: Demo options...")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Launch the interactive demo:")
    print("   streamlit run scripts/demo.py")
    print("\n2. Explore the Jupyter notebook:")
    print("   jupyter notebook notebooks/analysis.ipynb")
    print("\n3. View generated data:")
    print("   ls -la data/")
    print("\n4. View model results:")
    print("   ls -la models/")
    print("\n5. Run specific tests:")
    print("   python -m pytest tests/test_cold_start.py -v")
    
    # Ask if user wants to launch demo
    try:
        launch_demo = input("\nWould you like to launch the Streamlit demo now? (y/n): ").lower().strip()
        if launch_demo in ['y', 'yes']:
            logger.info("Launching Streamlit demo...")
            logger.info("Demo will be available at: http://localhost:8501")
            logger.info("Press Ctrl+C to stop the demo")
            subprocess.run("streamlit run scripts/demo.py", shell=True)
    except KeyboardInterrupt:
        logger.info("Demo launch cancelled")
    except Exception as e:
        logger.warning(f"Could not launch demo: {e}")
    
    logger.info("Pipeline completed!")

if __name__ == "__main__":
    main()
