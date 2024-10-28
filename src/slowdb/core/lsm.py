from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os
import shutil
import time
from collections import defaultdict

class SSTable:
    """Sorted String Table implementation."""
    
    def __init__(self, base_path: Path, level: int, table_id: int):
        self.base_path = base_path
        self.level = level
        self.table_id = table_id
        self.file_path = base_path / f"L{level}-{table_id}.sst"
        self.index: Dict[str, int] = {}
    
    def write(self, data: Dict[str, Any]) -> None:
        """Write sorted data to SSTable file."""
        with open(self.file_path, 'w') as f:
            # Write data section
            offset = 0
            for key, value in sorted(data.items()):
                entry = json.dumps({key: value}) + '\n'
                f.write(entry)
                self.index[key] = offset
                offset += len(entry.encode())
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value for key from SSTable."""
        if key not in self.index:
            return None
            
        with open(self.file_path, 'r') as f:
            f.seek(self.index[key])
            line = f.readline()
            entry = json.loads(line)
            return entry[key]

class LSMTree:
    """Log-Structured Merge Tree implementation."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.memtable = {}
        self.immutable_memtables = []
        self.levels = defaultdict(list)
        self.memtable_size_limit = 1000
        self._load_existing_tables()

    def _load_existing_tables(self) -> None:
        """Load existing SSTables from disk."""
        for level in range(self.max_level):
            self.levels[level] = []
            pattern = f"L{level}-*.sst"
            for path in self.base_path.glob(pattern):
                table_id = int(path.stem.split('-')[1])
                sstable = SSTable(self.base_path, level, table_id)
                self.levels[level].append(sstable)
    
    def put(self, key: str, value: Any) -> None:
        """Insert or update a key-value pair."""
        self.memtable[key] = value
        self._maybe_flush()
    
    def get(self, key: str) -> Any:
        """Retrieve value for a key."""
        # Check memtable
        if key in self.memtable:
            return self.memtable[key]
        
        # Check immutable memtables
        for table in self.immutable_memtables:
            if key in table:
                return table[key]
        
        # Check SSTables from newest to oldest
        for level in range(self.max_level):
            for sstable in reversed(self.levels[level]):
                value = sstable.get(key)
                if value is not None:
                    return value
        
        return None
    
    def _maybe_flush(self) -> None:
        """Check if memtable needs flushing and handle it."""
        if len(self.memtable) >= self.memtable_size_limit:
            self.immutable_memtables.append(self.memtable)
            self.memtable = {}
            self._compact_immutable_memtables()
    
    def _compact_immutable_memtables(self) -> None:
        """Compact immutable memtables into Level 0 SSTable."""
        if not self.immutable_memtables:
            return
            
        # Merge all immutable memtables
        merged_data = {}
        for table in self.immutable_memtables:
            merged_data.update(table)
        
        # Create new SSTable
        table_id = int(time.time() * 1000)
        sstable = SSTable(self.base_path, 0, table_id)
        sstable.write(merged_data)
        
        # Add to level 0 and clear immutable memtables
        if 0 not in self.levels:
            self.levels[0] = []
        self.levels[0].append(sstable)
        self.immutable_memtables.clear()
        
        # Check if we need to compact this level
        self._maybe_compact_level(0)
    
    def _maybe_compact_level(self, level: int) -> None:
        """Compact level if it has too many SSTables."""
        max_tables = 4 ** level
        if len(self.levels[level]) > max_tables:
            self._compact_level(level)
    
    def _compact_level(self, level: int) -> None:
        """Compact a level by merging its tables with the next level."""
        tables = self.levels.get(level, [])
        if not tables:
            return

        # Create new merged table
        merged_data = {}
        for table in tables:
            merged_data.update(table.data)

        if merged_data:
            new_table = SSTable(self.base_path, level + 1)
            new_table.data = merged_data
            
            # Ensure directory exists
            new_table.file_path.parent.mkdir(parents=True, exist_ok=True)
            new_table.flush()

            # Delete old tables after successful merge
            for table in tables:
                try:
                    if table.file_path.exists():
                        table.file_path.unlink()
                except (FileNotFoundError, PermissionError):
                    pass  # Ignore file operation errors

            # Update levels
            self.levels[level] = []
            if level + 1 not in self.levels:
                self.levels[level + 1] = []
            self.levels[level + 1].append(new_table)
