import pytest
import os
from pathlib import Path
from slowdb.core.storage import SegmentFile

@pytest.fixture
def temp_segment_path(tmp_path):
    return tmp_path / "test_segment.db"

@pytest.fixture
def segment_file(temp_segment_path):
    return SegmentFile(temp_segment_path, create=True)

def test_segment_creation(segment_file, temp_segment_path):
    assert os.path.exists(temp_segment_path)
    assert segment_file.mmap is not None

def test_append_and_read(segment_file):
    test_data = b"Hello, World!"
    offset = segment_file.append(test_data)
    
    # Read back data
    read_data = segment_file.read(offset, len(test_data))
    assert read_data == test_data

def test_multiple_appends(segment_file):
    data1 = b"First data"
    data2 = b"Second data"
    
    offset1 = segment_file.append(data1)
    offset2 = segment_file.append(data2)
    
    assert segment_file.read(offset1, len(data1)) == data1
    assert segment_file.read(offset2, len(data2)) == data2

def test_segment_close(segment_file):
    segment_file.close()
    assert segment_file.mmap.closed
    assert segment_file.file.closed

def test_invalid_read(segment_file):
    # Reading beyond file size should return empty bytes
    result = segment_file.read(1000, 10)
    assert result == b''

def test_segment_reopen(temp_segment_path):
    # Create and write data
    segment1 = SegmentFile(temp_segment_path, create=True)
    test_data = b"Test data"
    offset = segment1.append(test_data)
    segment1.close()
    
    # Reopen and verify data
    segment2 = SegmentFile(temp_segment_path, create=False)
    assert segment2.read(offset, len(test_data)) == test_data
