import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
import heapq
from dataclasses import dataclass
from .metrics import DistanceMetric

@dataclass
class SearchResult:
    """Search result containing node id and distance."""
    id: str
    distance: float
    
    def __lt__(self, other):
        return self.distance < other.distance

class HNSWGraph:
    """Hierarchical Navigable Small World graph implementation."""
    
    def __init__(self, 
                 dim: int,
                 M: int = 16,           # Max neighbors per node
                 ef_construction: int = 200,  # Size of candidate list during construction
                 ml_max: int = 16,      # Max layer for any element
                 metric: str = 'euclidean'):
        
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # Max neighbors for layer 0
        self.ef_construction = ef_construction
        self.ml_max = ml_max
        self.metric = DistanceMetric(metric)
        
        # Core data structures
        self.nodes: Dict[str, np.ndarray] = {}  # id -> vector
        self.layers: Dict[int, Set[str]] = {}   # layer -> node ids
        self.neighbors: Dict[str, Dict[int, Set[str]]] = {}  # id -> layer -> neighbor ids
        self.entry_point: Optional[str] = None
        self.max_layer = 0
        
    def _get_random_level(self) -> int:
        """Generate random level with exponential decay."""
        return int(-np.log(np.random.random()) * self.M)
    
    def _select_neighbors(self, 
                         candidates: List[SearchResult], 
                         M: int,
                         layer: int) -> List[str]:
        """Select best M neighbors from candidates."""
        # Simple greedy selection
        return [c.id for c in sorted(candidates)[:M]]
    
    def _search_layer(self, 
                     query: np.ndarray, 
                     entry_point: str,
                     ef: int,
                     layer: int) -> List[SearchResult]:
        """Search for nearest neighbors in a single layer."""
        visited = {entry_point}
        candidates = []
        results = []
        
        # Initialize with entry point
        dist = self.metric(query, self.nodes[entry_point])
        entry_result = SearchResult(entry_point, dist)
        heapq.heappush(candidates, entry_result)
        heapq.heappush(results, entry_result)
        
        while candidates:
            current = heapq.heappop(candidates)
            furthest_dist = results[-1].distance if len(results) >= ef else float('inf')
            
            if current.distance > furthest_dist:
                break
            
            # Check neighbors of current node
            for neighbor_id in self.neighbors[current.id].get(layer, set()):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = self.metric(query, self.nodes[neighbor_id])
                    
                    if len(results) < ef or dist < furthest_dist:
                        neighbor_result = SearchResult(neighbor_id, dist)
                        heapq.heappush(candidates, neighbor_result)
                        heapq.heappush(results, neighbor_result)
                        
                        if len(results) > ef:
                            heapq.heappop(results)
        
        return sorted(results)
    
    def insert(self, id: str, vector: np.ndarray) -> None:
        """Insert a new vector into the index."""
        if id in self.nodes:
            raise ValueError(f"Node {id} already exists in the index")
        
        self.nodes[id] = vector
        
        # Generate random level
        level = min(self._get_random_level(), self.ml_max)
        
        # Initialize node's neighbor structure
        self.neighbors[id] = {}
        for l in range(level + 1):
            if l not in self.layers:
                self.layers[l] = set()
            self.layers[l].add(id)
            self.neighbors[id][l] = set()
        
        # Handle first insertion
        if self.entry_point is None:
            self.entry_point = id
            self.max_layer = level
            return
        
        # Find entry point for insertion
        curr = self.entry_point
        curr_dist = self.metric(vector, self.nodes[curr])
        
        # Search from top to bottom
        for l in range(self.max_layer, -1, -1):
            changed = True
            while changed:
                changed = False
                
                # Try to find better candidates
                for neighbor_id in self.neighbors[curr].get(l, set()):
                    dist = self.metric(vector, self.nodes[neighbor_id])
                    if dist < curr_dist:
                        curr = neighbor_id
                        curr_dist = dist
                        changed = True
            
            # Connect at appropriate layers
            if l <= level:
                candidates = self._search_layer(vector, curr, self.ef_construction, l)
                neighbors = self._select_neighbors(candidates, self.M_max0 if l == 0 else self.M, l)
                
                # Add bidirectional connections
                for neighbor_id in neighbors:
                    self.neighbors[id][l].add(neighbor_id)
                    self.neighbors[neighbor_id][l].add(id)
        
        # Update entry point if necessary
        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = id
    
    def search(self, query: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if not self.entry_point:
            return []
        
        curr = self.entry_point
        
        # Search from top to bottom
        for l in range(self.max_layer, 0, -1):
            results = self._search_layer(query, curr, 1, l)
            curr = results[0].id
        
        # Search bottom layer with ef = k
        results = self._search_layer(query, curr, k, 0)
        return [(r.id, r.distance) for r in results[:k]]

