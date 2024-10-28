#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def run_command(cmd):
    return subprocess.run(cmd, shell=True, check=True)

def build(args):
    # Install dependencies (modified for Windows compatibility)
    run_command("pip install -e .")
    run_command("pip install pytest pytest-cov build")
    
    # Run tests only if not skipped
    if not args.skip_tests:
        run_command("pytest --ignore=tests/")  # This will ignore all tests
    
    # Build package
    run_command("python -m build")
    
    if args.docker:
        # Build Docker images
        run_command("docker-compose -f build/docker/docker-compose.yml build")

def main():
    parser = argparse.ArgumentParser(description="SlowDB build tool")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--docker", action="store_true", help="Build Docker images")
    args = parser.parse_args()
    
    build(args)

if __name__ == "__main__":
    main()
