# Using SlowDB


SlowDB is an educational vector database implementation designed for efficient storage and retrieval of vector data. This guide will help you integrate SlowDB into your project.

# Installation

After building the package with `make.py`, you'll find the `.whl` file in the `dist/` directory. Install it using pip:

Clone the repository:

```bash
   git clone https://github.com/yourusername/slowdb.git
   cd slowdb
```

If you have the `.whl` file, install it using pip:
```bash
pip install your_package.whl
```
Verify the installation:
```bash
pip list
```

# Use the package

In your code, import the package and use it as follows:

```python
from pathlib import Path
from slowdb.core.vector_store import VectorStorage

# Define the storage path and vector dimension
storage_path = Path("path/to/storage") # Path to the storage directory
dimension = 128  # Example dimension for your vectors

# Initialize the vector storage
vector_storage = VectorStorage(dimension=dimension, storage_path=storage_path)
```

## Basic usage

Store a vector:

```python
import numpy as np

# Create a sample vector
vector_id = "vector_1"
vector = np.random.rand(dimension)  # Random vector of specified dimension

# Store the vector
vector_storage.store_vector(vector_id, vector)
```

Retrieve a vector:

```python
# Retrieve the vector
retrieved_vector = vector_storage.get_vector(vector_id)

if retrieved_vector is not None:
    print("Retrieved vector:", retrieved_vector)
else:
    print("Vector not found.")
```

Train the compressor:

```python
# Create a list of vectors for training
training_vectors = [np.random.rand(dimension) for _ in range(100)]

# Train the compressor
vector_storage.force_train_compression(training_vectors)
```

# To Dockerize the application, run the following approach:

```Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire SlowDB project into the container
COPY . .

# Expose any ports if necessary (e.g., for a web server)
# EXPOSE 8000

# Command to run the application (modify as needed)
CMD ["python", "src/slowdb/__init__.py"]
```

Create a requirements.txt file:

```txt
numpy
scikit-learn
```

Create a docker-compose.yml file:
```yml
version: '3.8'

services:
  slowdb:
    build: .
    volumes:
      - .:/app  # Mount the current directory to /app in the container
    ports:
      - "8000:8000"  # Map port 8000 on the host to port 8000 in the container
    environment:
      - PYTHONUNBUFFERED=1  # Ensure logs are printed in real-time
```

Build and run the container:

```bash
docker-compose build && docker-compose up
```