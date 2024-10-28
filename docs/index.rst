SlowDB Documentation
==================

SlowDB is an educational vector database implementation focused on learning database internals,
vector storage, and search mechanisms.

Core Components
-------------

Vector Storage
~~~~~~~~~~~~

The vector storage system consists of three main components:

1. **VectorStorage**: Main interface for storing and retrieving vectors
2. **VectorCompressor**: Handles vector compression using Product Quantization
3. **VectorCompactor**: Manages segment compaction and storage optimization

.. code-block:: python

    from slowdb.core.vector_store import VectorStorage
    
    # Initialize storage
    storage = VectorStorage(dimension=128, storage_path="path/to/storage")
    
    # Store vector
    vector_id = "vector1"
    vector = np.random.random(128)
    storage.store_vector(vector_id, vector)
    
    # Retrieve vector
    retrieved = storage.get_vector(vector_id)

Vector Compression
~~~~~~~~~~~~~~~

The system uses Product Quantization for efficient vector compression:

.. code-block:: python

    # Train compression on sample vectors
    sample_vectors = [np.random.random(128) for _ in range(1000)]
    storage.train_compression(sample_vectors)
    
    # Vectors are automatically compressed during storage
    storage.store_vector("compressed_vector", vector)

Storage Management
~~~~~~~~~~~~~~~

Automatic segment compaction helps maintain storage efficiency:

.. code-block:: python

    # Compact segments when too many exist
    storage.maybe_compact_segments(threshold=5)

API Reference
-----------

.. autoclass:: slowdb.core.vector_store.VectorStorage
    :members:
    :undoc-members:

.. autoclass:: slowdb.core.vector_store.VectorCompressor
    :members:
    :undoc-members:

.. autoclass:: slowdb.core.vector_store.VectorCompactor
    :members:
    :undoc-members:

HNSW Index
---------

The Hierarchical Navigable Small World (HNSW) index is a graph-based structure for approximate nearest neighbor search.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from slowdb.index.hnsw import HNSWGraph
    import numpy as np

    # Initialize index
    index = HNSWGraph(
        dim=128,          # Vector dimension
        M=16,             # Max neighbors per node
        ef_construction=200,  # Construction-time candidate pool size
        ml_max=16,        # Max layer for any element
        metric='euclidean'  # Distance metric
    )

    # Insert vectors
    vector = np.random.random(128)
    index.insert("doc1", vector)

    # Search for nearest neighbors
    query = np.random.random(128)
    results = index.search(query, k=5)  # Get top 5 nearest neighbors

Parameters
~~~~~~~~~

- **dim**: Dimension of vectors
- **M**: Maximum number of connections per node (except layer 0)
- **ef_construction**: Size of dynamic candidate list during construction
- **ml_max**: Maximum layer for any element
- **metric**: Distance metric ('euclidean', 'cosine', 'manhattan', 'dot')

Implementation Details
~~~~~~~~~~~~~~~~~~~

The HNSW index uses a layered graph structure where:

1. Each node is assigned a random level
2. Higher layers form a coarse graph for quick traversal
3. Lower layers provide more precise connections
4. Layer 0 contains all nodes with more connections (M_max0 = 2*M)

Search Process
~~~~~~~~~~~~

1. Start at entry point in top layer
2. Descend through layers using greedy search
3. Perform beam search in bottom layer
4. Return k nearest neighbors

API Reference
~~~~~~~~~~~

.. autoclass:: slowdb.index.hnsw.HNSWGraph
    :members:
    :undoc-members:
    :special-members: __init__
