#!/usr/bin/env python3
"""
Build and publish script for MLEX package
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def clean_build():
    """Clean previous build artifacts"""
    print("Cleaning previous build artifacts...")
    run_command("rm -rf build/ dist/ *.egg-info/")
    run_command("find . -name '*.pyc' -delete")
    run_command("find . -name '__pycache__' -type d -exec rm -rf {} +")


def build_package():
    """Build the package"""
    print("Building package...")
    run_command("python -m build")


def check_package():
    """Check the built package"""
    print("Checking package...")
    run_command("python -m twine check dist/*")


def upload_to_testpypi():
    """Upload to TestPyPI"""
    print("Uploading to TestPyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")


def upload_to_pypi():
    """Upload to PyPI"""
    print("Uploading to PyPI...")
    run_command("python -m twine upload dist/*")


def install_dev_dependencies():
    """Install development dependencies"""
    print("Installing development dependencies...")
    run_command("pip install build twine")


def main():
    parser = argparse.ArgumentParser(description="Build and publish MLEX package")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI instead of PyPI")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before building")
    parser.add_argument("--check", action="store_true", help="Only check the package, don't upload")
    parser.add_argument("--install-deps", action="store_true", help="Install development dependencies")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dev_dependencies()
        return
    
    if args.clean:
        clean_build()
    
    build_package()
    check_package()
    
    if args.check:
        print("Package check completed successfully!")
        return
    
    if args.test:
        upload_to_testpypi()
        print("Package uploaded to TestPyPI successfully!")
    else:
        upload_to_pypi()
        print("Package uploaded to PyPI successfully!")


if __name__ == "__main__":
    main() 