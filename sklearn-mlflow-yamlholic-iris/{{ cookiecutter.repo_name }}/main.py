import sys
from src.{{ cookiecutter.python_package }}.run import run_package


sys.path.append("src")

if __name__ == "__main__":
    run_package()
