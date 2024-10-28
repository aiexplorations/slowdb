import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from .storage import SegmentFile
from .lsm import LSMTree

class VectorStorage:
    """Handles vector data storage and basic quantization."""
    
    def __init__(self, dimension: int, storage_path: Path, training_threshold: int = 100):
        self.dimension = dimension
        self.storage_path = storage_path
        self.training_threshold = training_threshold
        self.training_vectors = []  # Buffer for training vectors
        self.active_segment = None
        self.lsm_tree = LSMTree(base_path=storage_path)
        # Reduce clusters for testing
        self.compressor = VectorCompressor(dimension, n_clusters=8)  # Much smaller for testing
        self.compactor = VectorCompactor(storage_path)
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory and active segment."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._create_new_segment()
    
    def _create_new_segment(self) -> None:
        """Create a new active segment file."""
        segment_id = len(list(self.storage_path.glob("segment_*.db")))
        segment_path = self.storage_path / f"segment_{segment_id:06d}.db"
        self.active_segment = SegmentFile(segment_path, create=True)
    
    def train_compression(self, vectors: List[np.ndarray]) -> None:
        """Train the vector compressor on a set of vectors."""
        if len(vectors) < self.compressor.n_clusters:
            raise ValueError(f"Need at least {self.compressor.n_clusters} training vectors")
        self.compressor.train(vectors)
    
    def store_vector(self, vector_id: str, vector: np.ndarray) -> None:
        """Store a vector with its ID."""
        if len(vector) != self.dimension:
            vector = vector[:self.dimension]  # Truncate or pad as needed
        
        vector_bytes = vector.astype(np.float64).tobytes()
        if not self.active_segment:
            self.active_segment = SegmentFile.create(self.storage_path)
        
        offset = self.active_segment.write(vector_bytes)
        self.lsm_tree.put(vector_id, {
            "segment_id": self.active_segment.file_name,
            "offset": offset,
            "size": len(vector_bytes),
            "compressed": False
        })
    
    def _append_vector(self, vector: np.ndarray) -> None:
        """Append vector data to active segment."""
        vector_bytes = vector.tobytes()
        self.active_segment.append(vector_bytes)
    
    def _get_current_offset(self) -> int:
        """Get current offset in active segment."""
        return self.active_segment.mmap.tell()
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Retrieve a vector by its ID."""
        metadata = self.lsm_tree.get(vector_id)
        if metadata is None:
            return None
        
        try:
            segment = SegmentFile(self.storage_path / metadata["segment_id"])
            vector_bytes = segment.read(metadata["offset"], metadata["size"])
            
            if metadata.get("compressed", False):
                vector = self.compressor.decompress(vector_bytes)
            else:
                vector = np.frombuffer(vector_bytes, dtype=np.float64)
            
            return vector[:self.dimension]  # Ensure correct dimensionality
        except Exception as e:
            print(f"Error retrieving vector: {e}")
            return None
    
    def maybe_compact_segments(self, threshold: int = 5) -> None:
        """Compact segments if there are too many."""
        segments = list(self.storage_path.glob("segment_*.db"))
        if len(segments) > threshold:
            segment_files = [SegmentFile(path) for path in segments]
            new_segment, offset_map = self.compactor.compact_segments(segment_files)
            
            # Update metadata in LSM tree
            for vector_id, new_offset in offset_map.items():
                metadata = self.lsm_tree.get(vector_id)
                if metadata:
                    metadata["segment_id"] = new_segment.path.name
                    metadata["offset"] = new_offset
                    self.lsm_tree.put(vector_id, metadata)

    # Add method for testing purposes
    def force_train_compression(self, vectors: List[np.ndarray]) -> None:
        """Force immediate training of compressor (useful for testing)."""
        self.compressor.train(vectors)
        self.training_vectors = []  # Clear any buffered vectors

class VectorCompressor:
    """Handles vector compression using Product Quantization."""
    
    def __init__(self, dimension: int, n_subvectors: int = None, n_clusters: int = 16):
        # Automatically choose n_subvectors if not provided
        if n_subvectors is None:
            # Find largest factor of dimension that's <= 4
            n_subvectors = min(2, dimension)
            while dimension % n_subvectors != 0 and n_subvectors > 1:
                n_subvectors -= 1
            
        if dimension % n_subvectors != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by n_subvectors {n_subvectors}")
            
        self.dimension = dimension
        self.n_subvectors = n_subvectors
        self.subvector_dim = dimension // n_subvectors
        self.n_clusters = n_clusters
        self.codebooks = None
        self.is_trained = False
        self.min_vals = None  # For normalization
        self.max_vals = None  # For normalization
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to improve clustering."""
        if self.min_vals is None:
            self.min_vals = np.min(vectors, axis=0)
            self.max_vals = np.max(vectors, axis=0)
        
        denominator = (self.max_vals - self.min_vals)
        denominator[denominator == 0] = 1  # Avoid division by zero
        return (vectors - self.min_vals) / denominator
    
    def _denormalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return vector * (self.max_vals - self.min_vals) + self.min_vals
    
    def train(self, vectors: List[np.ndarray]) -> None:
        """Train the quantizer on a set of vectors."""
        if len(vectors) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} training vectors")
            
        vectors = np.array(vectors)
        normalized_vectors = self._normalize_vectors(vectors)
        
        from sklearn.cluster import MiniBatchKMeans
        
        self.codebooks = []
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            subvectors = normalized_vectors[:, start_idx:end_idx]
            
            # Use MiniBatchKMeans for better performance on large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=min(1000, len(vectors)),
                init='k-means++',
                random_state=42
            )
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_)
        
        self.is_trained = True
    
    def compress(self, vector: np.ndarray) -> bytes:
        """Compress a vector using trained codebooks."""
        if not self.is_trained:
            raise RuntimeError("Compressor must be trained before use. Call train() first.")
        
        normalized_vector = self._normalize_vectors(vector.reshape(1, -1))[0]
        codes = []
        
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            subvector = normalized_vector[start_idx:end_idx]
            
            # Find nearest centroid using efficient broadcasting
            distances = np.sum((self.codebooks[i] - subvector) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            codes.append(nearest_idx)
        
        return np.array(codes, dtype=np.uint8).tobytes()
    
    def decompress(self, codes: bytes) -> np.ndarray:
        """Decompress vector from codes."""
        if not self.is_trained:
            raise RuntimeError("Compressor must be trained before decompression")
            
        indices = np.frombuffer(codes, dtype=np.uint8)
        normalized_vector = np.zeros(self.dimension)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            normalized_vector[start_idx:end_idx] = self.codebooks[i][indices[i]]
        
        return self._denormalize_vector(normalized_vector)

class VectorCompactor:
    """Handles vector segment compaction."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
    
    def compact_segments(self, segments: List[SegmentFile]) -> SegmentFile:
        """Merge multiple segments into a single compact segment."""
        # Create new segment for compacted data
        new_segment_id = len(list(self.storage_path.glob("segment_*.db")))
        new_segment_path = self.storage_path / f"segment_{new_segment_id:06d}.db"
        new_segment = SegmentFile(new_segment_path, create=True)
        
        # Copy valid vectors to new segment
        offset_map = {}
        for segment in segments:
            # Read metadata and copy valid vectors
            with open(segment.path, 'rb') as f:
                while True:
                    try:
                        metadata = segment.read_metadata()
                        if not metadata:
                            break
                        
                        vector_data = segment.read_vector(metadata['offset'], metadata['size'])
                        new_offset = new_segment.append(vector_data)
                        offset_map[metadata['id']] = new_offset
                    except EOFError:
                        break
        
        # Clean up old segments
        for segment in segments:
            segment.close()
            segment.path.unlink()
        
        return new_segment, offset_map
