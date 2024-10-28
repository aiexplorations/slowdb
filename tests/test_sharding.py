import pytest
import numpy as np
from pathlib import Path
from slowdb.dist.sharding import ShardManager, Shard

@pytest.fixture
def temp_storage_path(tmp_path):
    return tmp_path / "shard_test"

@pytest.fixture
def shard_manager(temp_storage_path):
    return ShardManager(
        node_id="node1",
        storage_path=temp_storage_path,
        dimension=10,
        num_shards=4,
        virtual_nodes=2
    )

def test_initialization(shard_manager):
    assert shard_manager.num_shards == 4
    assert shard_manager.virtual_nodes == 2
    assert len(shard_manager.hash_ring) == 8  # num_shards * virtual_nodes

@pytest.mark.skip(reason="Sharding tests temporarily disabled")
def test_shard_assignment(shard_manager):
    # Initialize shards before testing
    shard_manager.initialize_shards(['node1'])  # Ensure shards are initialized

    # Add test vectors
    vectors = [(f"vec_{i}", np.random.random(10)) for i in range(10)]
    for vec_id, vector in vectors:
        shard_manager.store_vector(vec_id, vector)  # Ensure vectors are stored

    # Get shard assignments
    assignments = [shard_manager._get_shard(vid) for vid, _ in vectors]
    assert all(a is not None for a in assignments)  # Verify all vectors are assigned to shards
    # Verify consistent hashing property
    shard1 = shard_manager._get_shard("vec_1")
    shard2 = shard_manager._get_shard("vec_1")
    assert shard1.shard_id == shard2.shard_id

@pytest.mark.skip(reason="Sharding tests temporarily disabled")
def test_node_addition(shard_manager):
    # Add initial vectors
    for i in range(5):
        shard_manager.store_vector(f"vec_{i}", np.random.random(10))

    # Add new node
    moved_shards = shard_manager.add_node("node2")
    assert len(moved_shards) > 0  # Verify shards were redistributed
