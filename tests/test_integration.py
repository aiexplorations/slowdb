import pytest
import numpy as np
from pathlib import Path
from slowdb.core.vector_store import VectorStorage
from slowdb.index.hnsw import HNSWGraph

@pytest.fixture
def test_db(tmp_path):
    storage = VectorStorage(dimension=10, storage_path=tmp_path / "vectors")
    index = HNSWGraph(dim=10, M=8, ef_construction=64)
    return storage, index

@pytest.fixture
def sample_vectors():
    np.random.seed(42)
    return [
        (f"vec_{i}", np.random.random(10))
        for i in range(20)
    ]

@pytest.mark.skip(reason="Integration tests temporarily disabled")
def test_basic_workflow(test_db, sample_vectors):
    storage, index = test_db
    
    # Store vectors and build index
    for vec_id, vector in sample_vectors:
        assert storage.store_vector(vec_id, vector) is not None  # Ensure storage is successful
        index.insert(vec_id, vector)
    
    # Test retrieval
    vec_id, original = sample_vectors[0]
    retrieved = storage.get_vector(vec_id)
    assert retrieved is not None  # Ensure retrieval is successful
    assert np.allclose(original, retrieved)
    
    # Test search
    query = np.random.random(10)
    results = index.search(query, k=5)
    assert len(results) == 5

@pytest.mark.skip(reason="Integration tests temporarily disabled")
def test_persistence(tmp_path, sample_vectors):
    # Create and populate DB
    storage = VectorStorage(dimension=10, storage_path=tmp_path / "vectors")
    for vec_id, vector in sample_vectors[:5]:
        assert storage.store_vector(vec_id, vector) is not None  # Ensure storage is successful
    storage.active_segment.close()

    # Reopen and verify
    new_storage = VectorStorage(dimension=10, storage_path=tmp_path / "vectors")
    vec_id, original = sample_vectors[0]
    retrieved = new_storage.get_vector(vec_id)
    assert retrieved is not None  # Ensure retrieval is successful
    assert np.allclose(original, retrieved)
