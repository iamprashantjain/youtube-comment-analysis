import os
from pathlib import Path  # Importing Path to create OS-compatible paths
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of files and directories to create
list_of_files = [
    ".github/workflows/.gitkeep",          # GitHub Actions workflows directory
    ".github/workflows/cicd.yaml",         # cicd pipeline
    "experiment/experiments.ipynb",        # Jupyter Notebook for experimentation
    "src/__init__.py",                      # Init file for the src package
    
    
    #api
    "app/__init__.py",
    "app/app.py",
    
    
    #aws deploy files
    "deploy/scripts/install_dependencies.sh",
    "deploy/scripts/start_docker.sh.sh",
    "deploy/scripts/wait_for_health.sh",
    
    
    # Components for various stages of the project
    "src/components/__init__.py",          # Init file for components package
    "src/components/data_ingestion.py",    # Data ingestion script
    "src/components/data_transformation.py",# Data transformation script
    "src/components/model_trainer.py",     # Model training script
    "src/components/model_evaluation.py",  # Model evaluation script
    "src/components/promote_model.py",     
    "src/components/register_model.py",    
    
    
    # Utilities for common functions
    "src/utils/utils.py",                   # Utilities script
    "src/logger/logging.py",                # Logger for application logging
    "src/exception/exception.py",           # Custom exception handling

    # Test directories
    "test/unit/__init__.py",                # Init file for unit tests
    "test/test_model.py",                   # Init file for integration tests
    

    # Configuration and setup files
    "appspec.yml",
    "Dockerfile",
    "dvc.yaml",
    "params.yaml",
    "README.md",
    "init_setup.sh",                        # Shell script for initial setup
    "requirements_prod.txt",                # Prod dependencies file
    "requirements_dev.txt",                 # Development dependencies file
    "setup.py",                             # Setup script for installation
    "setup.cfg",                            # Configuration file for the package
    "pyproject.toml",                       # Build system requirements file
    "tox.ini",                              # Tox configuration file
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir:  # Check if there is a directory path
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    
    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            logging.info(f"Creating file: {filepath}")
            pass    #create an empty file


logging.info("Directories and files created successfully.")