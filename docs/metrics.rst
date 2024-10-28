Distance Metrics
==============

The metrics module provides optimized implementations of various distance metrics
for vector similarity search.

Available Metrics
---------------

- **Euclidean Distance**: Standard L2 distance metric
- **Cosine Distance**: Angle-based similarity metric
- **Manhattan Distance**: L1 distance metric
- **Dot Product Distance**: Raw similarity score
- **Angular Distance**: Normalized angle-based distance

Usage Examples
------------

Basic Usage:

.. code-block:: python

    from slowdb.index.metrics import DistanceMetric
    
    # Initialize metric
    metric = DistanceMetric('euclidean')
    
    # Compare vectors
    distance = metric(vector1, vector2)

Batch Processing:

.. code-block:: python

    from slowdb.index.metrics import batch_distance
    
    # Calculate distances between query and multiple vectors
    distances = batch_distance(metric, query_vector, vector_list)

Performance Considerations
-----------------------

- Use batch_distance for multiple comparisons
- Vectorized operations for improved performance
- Optimized implementations for common metrics
- Memory-efficient computations

API Reference
-----------

.. automodule:: slowdb.index.metrics
    :members:
    :undoc-members:
    :show-inheritance:
