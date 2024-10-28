import os
import pathlib

def create_directory_structure():
    # Project root structure
    directories = [
        "src/slowdb/core",
        "src/slowdb/index",
        "src/slowdb/dist",
        "src/slowdb/utils",
        "tests",
        "examples",
        "docs",
    ]

    # Create directories
    for directory in directories:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    # Create empty files
    files = [
        # Root level files
        "pyproject.toml",
        "setup.cfg",
        "README.md",
        "LICENSE",
        ".gitignore",
        
        # Source files
        "src/slowdb/__init__.py",
        "src/slowdb/core/__init__.py",
        "src/slowdb/core/storage.py",
        "src/slowdb/core/lsm.py",
        "src/slowdb/core/vector_store.py",
        "src/slowdb/index/__init__.py",
        "src/slowdb/index/hnsw.py",
        "src/slowdb/index/metrics.py",
        "src/slowdb/dist/__init__.py",
        "src/slowdb/dist/sharding.py",
        "src/slowdb/dist/replication.py",
        "src/slowdb/utils/__init__.py",
        "src/slowdb/utils/serialization.py",
        "src/slowdb/utils/metrics.py",
        
        # Test files
        "tests/__init__.py",
        "tests/test_storage.py",
        "tests/test_lsm.py",
        "tests/test_vector_store.py",
        
        # Example files
        "examples/basic_usage.py",
        "examples/distributed_setup.py",
        
        # Documentation files
        "docs/conf.py",
        "docs/index.rst",
    ]

    # Create empty files
    for file_path in files:
        pathlib.Path(file_path).touch(exist_ok=True)

    print("Project structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()