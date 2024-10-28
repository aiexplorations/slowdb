import pytest
import numpy as np
from slowdb.index.metrics import (
    DistanceMetric, 
    euclidean_distance, 
    cosine_distance, 
    manhattan_distance, 
    dot_product_distance,
    angular_distance,
    batch_distance
)

@pytest.fixture
def vectors():
    np.random.seed(42)
    return np.random.random((100, 10))

@pytest.fixture
def query_vector():
    return np.ones(10)

def test_euclidean_distance():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert euclidean_distance(v1, v2) == np.sqrt(2)

def test_cosine_distance():
    v1 = np.array([1, 0])
    v2 = np.array([1, 1])
    assert np.isclose(cosine_distance(v1, v2), 1 - 1/np.sqrt(2))

def test_manhattan_distance():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 1])
    assert manhattan_distance(v1, v2) == 3

def test_dot_product_distance():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    assert dot_product_distance(v1, v2) == -(1*4 + 2*5 + 3*6)

def test_batch_distance(vectors, query_vector):
    metric = DistanceMetric('euclidean')
    distances = batch_distance(metric, query_vector, vectors)
    assert len(distances) == len(vectors)
    assert all(d >= 0 for d in distances)

def test_invalid_metric():
    with pytest.raises(ValueError):
        DistanceMetric('invalid_metric')

def test_metric_equivalence(vectors, query_vector):
    metric = DistanceMetric('euclidean')
    batch_results = batch_distance(metric, query_vector, vectors)
    individual_results = [metric(query_vector, v) for v in vectors]
    assert np.allclose(batch_results, individual_results)
