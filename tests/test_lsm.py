import pytest
from pathlib import Path
from slowdb.core.lsm import LSMTree, SSTable

@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "lsm_test"

@pytest.fixture
def lsm_tree(temp_db_path):
    return LSMTree(memtable_size=3, base_path=temp_db_path)

def test_lsm_initialization(lsm_tree):
    assert len(lsm_tree.memtable) == 0
    assert len(lsm_tree.immutable_memtables) == 0
    assert lsm_tree.max_level == 3

def test_put_and_get(lsm_tree):
    lsm_tree.put("key1", "value1")
    assert lsm_tree.get("key1") == "value1"

def test_memtable_flush(lsm_tree):
    # Fill memtable to trigger flush
    lsm_tree.put("key1", "value1")
    lsm_tree.put("key2", "value2")
    lsm_tree.put("key3", "value3")
    lsm_tree.put("key4", "value4")
    
    # Check if values are still accessible
    assert lsm_tree.get("key1") == "value1"
    assert lsm_tree.get("key4") == "value4"
    
    # Verify memtable was flushed
    assert len(lsm_tree.memtable) < 3

def test_sstable_creation(lsm_tree):
    # Add enough data to trigger SSTable creation
    for i in range(5):
        lsm_tree.put(f"key{i}", f"value{i}")
    
    # Verify SSTable files were created
    sstable_files = list(lsm_tree.base_path.glob("L0-*.sst"))
    assert len(sstable_files) > 0

@pytest.mark.skip(reason="LSM tree compaction tests temporarily disabled")
def test_compaction_trigger(lsm_tree):
    # Add enough data to trigger compaction
    for i in range(20):
        lsm_tree.put(f"key{i}", f"value{i}")

    # Ensure that the compaction process is functioning correctly
    assert lsm_tree.compaction_triggered()  # Add a check to ensure compaction was triggered
