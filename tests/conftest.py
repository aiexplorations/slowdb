import pytest
import numpy as np
from src.slowdb.core.vector_store import VectorStorage
from src.slowdb.dist.sharding import ShardManager

@pytest.fixture
def vector_storage(tmp_path):
    storage = VectorStorage(dimension=10, storage_path=tmp_path / "vectors", training_threshold=16)
    return storage

@pytest.fixture
def sample_vectors():
    # Create enough vectors for testing
    return [np.random.random(10) for _ in range(16)]

@pytest.fixture
def shard_manager():
    manager = ShardManager()
    # Pre-initialize with two nodes for testing
    manager.shards = {"node1": [], "node2": []}
    return manager
