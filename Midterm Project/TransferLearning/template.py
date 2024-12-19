import os 
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO,format ="[%(asctime)s]:%(message)s:%(lineno)d:")

current_dir = Path.cwd()
project_name = 'cnnClassifier' 

# list of files
list_of_files = [
    "github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utlis/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]
# Print current working directory and home directory
current_dir = os.getcwd()
logging.info(f"Current working directory: {current_dir}")

for filepath in list_of_files:
    filepath = current_dir / Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file: {filename}")
    
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File: {filename} already exists")

logging.info(f"Project setup completed in: {current_dir}")