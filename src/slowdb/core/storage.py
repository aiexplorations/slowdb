import mmap
import os
import struct
from pathlib import Path
from typing import Optional

class SegmentFile:
    """Represents a single segment file using memory-mapped I/O."""
    
    HEADER_FORMAT = "=Q"  # 8-byte unsigned long for segment size
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    def __init__(self, path: Path, create: bool = False):
        """Initialize a segment file for storing data.
        
        Args:
            path: Path to the segment file
            create: If True, create a new file. If False, open existing file.
        """
        self.path = path
        mode = 'w+b' if create else 'r+b'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Open or create the file
        self.file = open(path, mode)
        
        # Ensure file has at least 1 byte for memory mapping
        if create:
            self.file.write(struct.pack(self.HEADER_FORMAT, 0))
            self.file.flush()
        
        # Create memory map
        size = os.path.getsize(self.path)
        self.mmap = mmap.mmap(self.file.fileno(), size)
        self._size = size
    
    def append(self, data: bytes) -> int:
        """Append data to the segment file and return offset."""
        offset = self._size
        self.mmap.resize(self._size + len(data))
        self.mmap[offset:offset + len(data)] = data
        self._size += len(data)
        return offset
    
    def read(self, offset: int, size: int) -> bytes:
        """Read data from the segment file at given offset."""
        if offset >= self._size:
            return b''
        return self.mmap[offset:min(offset + size, self._size)]
    
    def close(self):
        """Close the segment file and memory mapping."""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
