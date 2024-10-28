Distribution Layer
================

The distribution layer in SlowDB handles data distribution across multiple nodes through replication and sharding.

Replication
----------

The replication module implements a leader-follower model for vector data replication.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from slowdb.dist.replication import ReplicationManager
    
    # Start leader node
    leader = ReplicationManager(
        node_id="leader",
        storage_path="/path/to/storage",
        dimension=128,
        port=8000
    )
    await leader.start_leader()
    
    # Start follower node
    follower = ReplicationManager(
        node_id="follower1",
        storage_path="/path/to/storage",
        dimension=128,
        port=8001
    )
    await follower.start_follower("localhost", 8000)

Configuration
~~~~~~~~~~~~

- **node_id**: Unique identifier for the node
- **storage_path**: Path for vector storage
- **dimension**: Vector dimension
- **host**: Host address (default: localhost)
- **port**: Port number for node communication

Sharding
-------

The sharding module implements consistent hashing for vector distribution across nodes.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from slowdb.dist.sharding import ShardManager
    
    # Initialize shard manager
    manager = ShardManager(
        node_id="node1",
        storage_path="/path/to/storage",
        dimension=128,
        num_shards=16,
        virtual_nodes=3
    )
    
    # Store vector (automatically routes to correct shard)
    await manager.store_vector("vector1", vector)

Configuration
~~~~~~~~~~~~

- **num_shards**: Number of shards (default: 16)
- **virtual_nodes**: Virtual nodes per physical node (default: 3)
- **dimension**: Vector dimension
- **storage_path**: Path for vector storage

Architecture
-----------

Leader-Follower Replication
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Leader handles all write operations
2. Changes are replicated to followers
3. Followers maintain read-only copies
4. Automatic failover support

Consistent Hashing
~~~~~~~~~~~~~~~~

1. Uses MurmurHash3 for vector distribution
2. Virtual nodes for better balance
3. Minimal redistribution on node changes
4. Automatic shard rebalancing

API Reference
-----------

.. automodule:: slowdb.dist.replication
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: slowdb.dist.sharding
    :members:
    :undoc-members:
    :show-inheritance:

Error Handling
------------

Common error scenarios and handling:

1. Node Failure
   - Automatic leader election
   - Replication catch-up
   - Shard redistribution

2. Network Issues
   - Retry mechanisms
   - Eventual consistency
   - Split-brain prevention

3. Data Consistency
   - Vector version tracking
   - Conflict resolution
   - Consistency levels

Performance Considerations
-----------------------

1. Replication
   - Async replication for better performance
   - Batched updates
   - Configurable consistency levels

2. Sharding
   - Virtual nodes for load balancing
   - Local shard preference
   - Minimal resharding on changes
