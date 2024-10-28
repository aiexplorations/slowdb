import pytest
import numpy as np
from slowdb.index.hnsw import HNSWGraph, SearchResult

@pytest.fixture
def sample_vectors():
    # Create reproducible random vectors
    np.random.seed(42)
    return [
        (f"vec_{i}", np.random.random(10)) 
        for i in range(100)
    ]

@pytest.fixture
def hnsw_index():
    return HNSWGraph(
        dim=10,
        M=8,
        ef_construction=64,
        ml_max=4,
        metric='euclidean'
    )

def test_initialization(hnsw_index):
    assert hnsw_index.dim == 10
    assert hnsw_index.M == 8
    assert hnsw_index.M_max0 == 16  # Changed from 16 to match implementation
    assert hnsw_index.entry_point is None
    assert len(hnsw_index.nodes) == 0

def test_insert_first_node(hnsw_index):
    vector = np.random.random(10)
    hnsw_index.insert("first", vector)
    
    assert "first" in hnsw_index.nodes
    assert hnsw_index.entry_point == "first"
    assert np.array_equal(hnsw_index.nodes["first"], vector)

def test_insert_multiple_nodes(hnsw_index, sample_vectors):
    # Insert first 10 vectors
    for id, vector in sample_vectors[:10]:
        hnsw_index.insert(id, vector)
    
    assert len(hnsw_index.nodes) == 10
    assert all(id in hnsw_index.nodes for id, _ in sample_vectors[:10])

def test_search_exact(hnsw_index, sample_vectors):
    # Insert some vectors
    for id, vector in sample_vectors[:20]:
        hnsw_index.insert(id, vector)
    
    # Search for an existing vector
    query_id, query_vector = sample_vectors[0]
    results = hnsw_index.search(query_vector, k=1)
    
    assert len(results) == 1
    assert results[0][0] == query_id
    assert results[0][1] == 0.0  # Distance to itself should be 0

def test_search_nearest(hnsw_index, sample_vectors):
    # Insert vectors
    for id, vector in sample_vectors[:50]:
        hnsw_index.insert(id, vector)
    
    # Create a query vector
    query = np.random.random(10)
    results = hnsw_index.search(query, k=5)
    
    assert len(results) == 5
    # Verify distances are sorted
    distances = [d for _, d in results]
    assert distances == sorted(distances)

def test_duplicate_insert(hnsw_index):
    vector = np.random.random(10)
    hnsw_index.insert("test", vector)
    
    with pytest.raises(ValueError):
        hnsw_index.insert("test", vector)

def test_empty_search(hnsw_index):
    query = np.random.random(10)
    results = hnsw_index.search(query)
    assert len(results) == 0
