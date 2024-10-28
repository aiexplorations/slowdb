import pytest
import asyncio
import numpy as np
from pathlib import Path
from aiohttp import web
from slowdb.dist.replication import ReplicationManager, ReplicationNode

@pytest.fixture
def temp_storage_path(tmp_path):
    return tmp_path / "replication_test"

@pytest.fixture
async def leader_node(temp_storage_path):
    manager = ReplicationManager(
        node_id="leader",
        storage_path=temp_storage_path / "leader",
        dimension=10,
        port=8000
    )
    await manager.start_leader()
    yield manager
    # Cleanup
    await manager.app.shutdown()

@pytest.fixture
async def follower_node(temp_storage_path):
    manager = ReplicationManager(
        node_id="follower1",
        storage_path=temp_storage_path / "follower1",
        dimension=10,
        port=8001
    )
    await manager.start_follower("localhost", 8000)
    yield manager
    # Cleanup
    await manager.app.shutdown()

async def test_leader_initialization(leader_node):
    assert leader_node.is_leader
    assert len(leader_node.followers) == 0
    assert leader_node.node_id == "leader"

async def test_follower_join(leader_node, follower_node):
    # Wait for follower to join
    await asyncio.sleep(0.1)
    assert "follower1" in leader_node.followers
    assert not follower_node.is_leader

async def test_vector_replication(leader_node, follower_node):
    vector = np.random.random(10)
    await leader_node.replicate_vector("test_vector", vector)
    
    # Wait for replication
    await asyncio.sleep(0.1)
    
    # Verify vector exists in both nodes
    assert np.array_equal(
        leader_node.vector_store.get_vector("test_vector"),
        follower_node.vector_store.get_vector("test_vector")
    )
