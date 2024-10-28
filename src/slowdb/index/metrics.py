"""
Vector Distance Metrics Module

Optimized implementations of various distance metrics for vector similarity search.
Includes batch processing capabilities and vectorized operations for performance.
"""

import numpy as np
from typing import Union, List, Callable
from functools import partial

class DistanceMetric:
    """Distance metric wrapper class that provides a unified interface for different metrics."""
    
    def __init__(self, metric_name: str = 'euclidean', **kwargs):
        """Initialize distance metric.
        
        Args:
            metric_name: Name of the metric to use ('euclidean', 'cosine', 'manhattan', 'dot')
            **kwargs: Additional parameters for specific metrics
        """
        self.metric_name = metric_name
        self.metric_func = self._get_metric_function(metric_name)
        self.kwargs = kwargs
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        return self.metric_func(x, y, **self.kwargs)
    
    def _get_metric_function(self, name: str) -> Callable:
        """Get the appropriate metric function based on name."""
        metrics = {
            'euclidean': euclidean_distance,
            'cosine': cosine_distance,
            'manhattan': manhattan_distance,
            'dot': dot_product_distance,
            'angular': angular_distance
        }
        
        if name not in metrics:
            raise ValueError(f"Unsupported metric: {name}. Available metrics: {list(metrics.keys())}")
        
        return metrics[name]

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean (L2) distance between two vectors.
    
    Optimized implementation using numpy's built-in operations.
    Distance = sqrt(sum((x - y)^2))
    """
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cosine distance between two vectors.
    
    Returns 1 - cosine_similarity for use as a distance metric.
    Distance = 1 - (dot(x,y) / (||x|| * ||y||))
    """
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1.0 - dot_product / (norm_x * norm_y)

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan (L1) distance between two vectors.
    
    Sum of absolute differences between vector components.
    Distance = sum(|x - y|)
    """
    return np.sum(np.abs(x - y))

def dot_product_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate negative dot product distance.
    
    Returns negative dot product to convert similarity to distance.
    Distance = -dot(x,y)
    """
    return -np.dot(x, y)

def angular_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate angular distance between two vectors.
    
    Converts cosine similarity to proper angular distance.
    Distance = arccos(cosine_similarity) / pi
    """
    similarity = 1.0 - cosine_distance(x, y)
    return np.arccos(min(1.0, max(-1.0, similarity))) / np.pi

def batch_distance(metric: DistanceMetric, 
                  query: np.ndarray, 
                  vectors: List[np.ndarray]) -> np.ndarray:
    """Calculate distances between query vector and multiple vectors efficiently.
    
    Args:
        metric: Distance metric to use
        query: Query vector
        vectors: List of vectors to compare against
    
    Returns:
        Array of distances
    """
    vectors = np.array(vectors)
    if metric.metric_name == 'euclidean':
        # Optimized euclidean distance calculation for batches
        return np.sqrt(np.sum((vectors - query) ** 2, axis=1))
    else:
        # Fallback for other metrics
        return np.array([metric(query, v) for v in vectors])

def euclidean_distance_vectorized(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean distance computation for batch processing."""
    return np.sqrt(np.sum((Y - x) ** 2, axis=1))

def cosine_distance_vectorized(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized cosine distance computation."""
    x_norm = np.linalg.norm(x)
    Y_norm = np.linalg.norm(Y, axis=1)
    dot_products = np.dot(Y, x)
    return 1.0 - dot_products / (x_norm * Y_norm)
