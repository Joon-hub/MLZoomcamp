import sys

def check_environment():
    if sys.prefix != sys.base_prefix:
        print(f"Running in a virtual environment: {sys.prefix}")
    else:
        print("Running in the system's default Python environment")

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

check_environment()