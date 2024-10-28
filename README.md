# slowdb
SlowDB is an educational vector database implementation focused on learning database internals, vector storage, and search mechanisms. Built entirely in Python for clarity and learning purposes, it prioritizes clear implementation over performance.

## Learning Objectives
- Understanding vector database architecture
- Implementing core database components from scratch
- Learning about LSM trees, HNSW indexes, and vector compression
- Exploring distributed systems concepts

## Components & Implementation Plan

### Phase 1: Core Storage
1. **Memory-Mapped Storage**
   - Basic file structure implementation: This allows for efficient file I/O operations by mapping files directly into memory, enabling faster access to data.
   - Simple append-only log: This design choice ensures that data is written sequentially, which is optimal for write-heavy workloads and simplifies recovery.
   - Memory mapping with Python's `mmap`: Using `mmap` allows the database to handle large files without loading them entirely into memory, thus optimizing memory usage.

2. **LSM Tree Implementation**
   - In-memory memtable using sorted dictionary: The memtable is a write-optimized structure that allows for fast inserts and updates. It is sorted to facilitate efficient reads and merges.
   - Simple SSTable implementation: SSTables (Sorted String Tables) are immutable data structures that store sorted key-value pairs, providing efficient read access and enabling easy compaction.
   - Basic compaction strategy: Compaction is crucial for maintaining performance as it merges smaller SSTables into larger ones, reducing the number of files and improving read efficiency.
   - Vector metadata storage: Metadata about vectors (e.g., dimensions, offsets) is stored alongside the actual data to facilitate quick access and management.

3. **Vector Storage**
   - Basic vector serialization: Vectors are serialized for storage, allowing for efficient disk usage and retrieval.
   - Simple product quantization implementation: This technique reduces the storage requirements of vectors while maintaining their representational quality, which is essential for large datasets.
   - NumPy-based vector operations: Leveraging NumPy for vector operations provides optimized performance due to its underlying C implementations.

### Phase 2: Search Capabilities
1. **HNSW Index**
   - Basic graph structure: The Hierarchical Navigable Small World (HNSW) graph is used for efficient nearest neighbor search, allowing for logarithmic time complexity in search operations.
   - Insertion algorithm: The insertion algorithm ensures that new vectors are added to the graph while maintaining its navigability and structure.
   - Search implementation: The search algorithm traverses the graph to find the nearest neighbors efficiently, utilizing a priority queue to manage candidate nodes.
   - Simple distance metrics: Various distance metrics (e.g., Euclidean, cosine) are implemented to measure similarity between vectors, allowing users to choose the most appropriate metric for their use case.

2. **Memory Management**
   - Active segment handling: This involves managing segments of data in memory to optimize read and write operations, ensuring that frequently accessed data is readily available.
   - Basic cache implementation: A caching layer is implemented to store recently accessed data, reducing the need for repeated disk I/O operations.
   - Memory usage monitoring: Monitoring memory usage helps in optimizing performance and preventing memory leaks, ensuring the system remains responsive.

### Phase 3: Distribution (Optional)
1. **Basic Distribution**
   - Consistent hashing implementation: This technique allows for efficient data distribution across multiple nodes, ensuring that the addition or removal of nodes has minimal impact on data locality.
   - Simple node management: Basic functionalities for managing nodes in a distributed system are implemented, allowing for easy scaling and maintenance.
   - Basic data partitioning: Data is partitioned across nodes to balance load and improve access times.

2. **Replication**
   - Leader-follower implementation: This model ensures data availability and fault tolerance by replicating data across multiple nodes, with one node acting as the leader and others as followers.
   - Basic failover mechanism: In case of node failure, the system can automatically promote a follower to a leader, ensuring continuous availability of the database.

## Technical Stack
- **Core Implementation**: Python 3.8+
- **Key Dependencies**:
  - NumPy (vector operations): Provides efficient array operations and mathematical functions.
  - mmap (memory-mapped files): Enables efficient file I/O operations.
  - Protocol Buffers (serialization): Used for efficient data serialization and communication.
  - gRPC (network communication): Facilitates communication between distributed components.

## Project Goals
- **Educational**: Clear, well-documented code over performance, making it easier for learners to understand the underlying concepts.
- **From Scratch**: Minimal use of specialized libraries to encourage learning and understanding of core principles.
- **Modular**: Clean interfaces between components to promote maintainability and extensibility.
- **Documented**: Extensive comments and documentation to aid understanding and facilitate contributions.
- **Testable**: Comprehensive test coverage to ensure reliability and correctness of the implementation.

## Non-Goals
- Production-level performance: The focus is on educational value rather than optimization for production use.
- Production-ready reliability: While reliability is considered, the primary goal is to provide a learning platform.
- Complex optimization techniques: The implementation prioritizes clarity over advanced optimizations.
- Advanced compression methods: Basic compression techniques are used to keep the implementation straightforward.

## Getting Started
(Coming soon)

## Development Roadmap
1. Basic storage engine implementation
2. Simple vector operations
3. HNSW index implementation
4. Basic query capabilities
5. Simple distribution system

## Contributing
This is an educational project. Contributions that improve code clarity, documentation, or add learning resources are welcome.
